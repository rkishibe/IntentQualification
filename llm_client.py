import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from utils import company_id

UNDERSTAND_QUERY_PROMPT = """You are a query understanding system that interprets a natural-language company search query and outputs a structured plan for filtering and ranking companies. The output must be a valid JSON object that conforms to the specified schema. Do not include any explanations or additional text outside of the JSON object.

{{
    "name": "submit_query_plan",
    "description": "Submit the structured interpretation of a company-search query.",
    "input_schema": {{
        "type": "object",
        "properties": {{
            "query_type": {{
                "type": "string",
                "enum": ["structured", "semantic", "hybrid", "ecosystem"],
                "description": "'structured': fully answered by explicit filters (location, size, revenue, industry, public status). 'semantic': needs judgment beyond raw fields (e.g. 'fast-growing', 'competing with traditional banks'). 'ecosystem': needs reasoning about a company's role in a supply chain or business relationship not stated directly (e.g. 'could supply packaging for a cosmetics brand'). 'hybrid': has both a hard filter and a judgment component."
            }},
            "structured_filters": {{
                "type": "object",
                "description": "Only include keys the query actually constrains.",
                "properties": {{
                    "country": {{"type": "string"}},
                    "employee_count_min": {{"type": "integer"}},
                    "employee_count_max": {{"type": "integer"}},
                    "revenue_min": {{"type": "number"}},
                    "revenue_max": {{"type": "number"}},
                    "year_founded_min": {{"type": "integer"}},
                    "year_founded_max": {{"type": "integer"}},
                    "is_public": {{"type": "boolean"}},
                    "naics_prefixes": {{
                        "type": "array",
                        "items": {{"type": "string"}},
                        "description": (
                            "NAICS prefixes for a literal industry match. Use the SHORTEST "
                            "prefix you are confident about -- prefer a 2-digit sector code "
                            "over guessing a longer, more specific code from memory. "
                            "Reference table:\\n"
                            "11 Agriculture/Forestry/Fishing, 21 Mining/Oil/Gas, 22 Utilities, "
                            "23 Construction, 31-33 Manufacturing, 42 Wholesale Trade, "
                            "44-45 Retail Trade, 48-49 Transportation/Warehousing/Logistics, "
                            "51 Information (software, IT, publishing), "
                            "52 Finance/Insurance (fintech, banking), "
                            "54 Professional/Scientific/Technical Services (consulting), "
                            "61 Educational Services, 62 Health Care, "
                            "3254 Pharmaceutical Manufacturing, "
                            "71 Arts/Entertainment, 72 Accommodation/Food Services, "
                            "81 Other Services.\\n"
                            "Leave EMPTY for role/ecosystem queries where the matching company "
                            "could plausibly be filed under an unrelated code -- rely on "
                            "expansion_terms and hypothetical_profile instead."
                        ),
                    }},
                }}
            }},
            "expansion_terms": {{
                "type": "array",
                "items": {{"type": "string"}},
                "description": "5-10 related concepts, synonyms, or roles to broaden retrieval."
            }},
            "hypothetical_profile": {{
                "type": "string",
                "description": "A short paragraph written as if it were the ideal matching company's own description -- used for HyDE-style embedding retrieval. For role/ecosystem queries, describe the SUPPLIER or SERVICE PROVIDER, not the end customer named in the query."
            }}
        }},
        "required": [
            "query_type",
            "structured_filters",
            "expansion_terms",
            "hypothetical_profile"
        ]
    }}
}}

User query:
{query}
"""

JUDGE_BATCH_PROMPT = """Query: {query}
Query type: {query_type}
What a genuine match looks like: {hypothetical_profile}

For EACH company below, decide whether it genuinely satisfies the query's intent -- not just
superficial keyword overlap with the query text. If the available fields are insufficient to
tell, say so honestly with low confidence rather than guessing match=true.

Respond with ONLY a JSON object of this exact shape, with one entry per company, in the same
order as given below. No explanation outside the JSON. No markdown code fences.

{{"verdicts": [
  {{"company_id": "...", "match": true, "confidence": 0.8, "reason": "one short sentence"}}
]}}

Companies:
{companies_json}"""

REGION_GROUPS = {
    "scandinavia": ["Sweden", "Norway", "Denmark"],
    "nordics": ["Sweden", "Norway", "Denmark", "Finland", "Iceland"],
    "europe": [
        "Austria", "Belgium", "Switzerland", "Germany", "Denmark", "Spain", "Finland", "France", "United Kingdom", "Greece", "Croatia",
        "Ireland", "Iceland", "Italy", "Latvia", "Luxembourg", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Sweden",
        "Ukraine",
    ],
}

class LocalLLMClient:
    """
    Free local LLM client for query understanding (Stage 0) and batched
    qualification (Stage 3).

    Like LocalEmbeddingClient above, the model is downloaded once from
    Hugging Face and then runs entirely locally -- no API key, no per-call
    cost, no rate limits. The tradeoff: a small open model can't be forced
    into a schema the way a hosted model's tool-use can, so this client
    leans on strict prompting plus a string-aware JSON extractor and a
    couple of retries, and degrades gracefully instead of crashing the
    pipeline if a response still can't be parsed.

    Swap `model_name` for a larger instruct model if you have a GPU and want
    more reliable JSON / better judgment quality, e.g. "Qwen/Qwen2.5-7B-Instruct"
    or "meta-llama/Llama-3.1-8B-Instruct". The default is small enough to run
    on CPU, at the cost of being less reliable on genuinely hard judgment calls.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 1024,
        device: torch.device = torch.device("xpu"),
        max_json_retries: int = 2,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_json_retries = max_json_retries
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
            device_map=self.device,
        )
        self.model.eval()

    # ---- low-level generation ----------------------------------------

    def _generate(self, prompt: str, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.device)

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _extract_json(text: str):
        """
        Pull the first well-formed JSON value out of a blob of LLM output
        that may include markdown fences, preamble, or trailing chatter.
        Bracket matching is string-aware so braces/brackets inside a JSON
        string value (e.g. inside a "reason" sentence) don't throw off the
        balance count.
        """
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        open_chars = {"{": "}", "[": "]"}
        start = None
        for i, ch in enumerate(text):
            if ch in open_chars:
                start = i
                break
        if start is None:
            return None

        stack = []
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in open_chars:
                stack.append(open_chars[ch])
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()
                    if not stack:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            return None
        return None

    def _generate_json(self, prompt: str, temperature: float = 0.0):
        current_prompt = prompt
        last_raw = None
        for _ in range(self.max_json_retries + 1):
            raw = self._generate(current_prompt, temperature=temperature)
            last_raw = raw
            parsed = self._extract_json(raw)
            if parsed is not None:
                return parsed
            current_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. Respond with"
                  " ONLY the JSON, no explanation, no markdown fences."
            )
        raise ValueError(
            f"Local model did not return parseable JSON after "
            f"{self.max_json_retries + 1} attempt(s). Last output: {last_raw!r}"
        )

    # ---- Stage 0: query understanding ----------------------------------

    def understand_query(self, query: str) -> dict:
        prompt = UNDERSTAND_QUERY_PROMPT.format(query=query)
        try:
            plan = self._generate_json(prompt, temperature=0.0)
        except ValueError:
            plan = {}

        if not isinstance(plan, dict):
            plan = {}

        # Defensive defaults: a bad/partial generation degrades to "hybrid"
        # (still runs embeddings + LLM judging) with no hard structured
        # filters, rather than crashing the whole query.
        plan.setdefault("query_type", "hybrid")
        plan.setdefault("structured_filters", {})
        plan.setdefault(
            "expansion_terms",
            [t for t in re.findall(r"[a-zA-Z]{4,}", query.lower())],
        )
        plan.setdefault("hypothetical_profile", f"A company that satisfies: {query}")
        return plan

    # ---- Stage 3: batched judging ---------------------------------------

    def judge_batch(
        self, query: str, plan: dict, companies: list, temperature: float = 0.0
    ) -> list:
        company_blocks = []
        for c in companies:
            cid = company_id({
                "website": c.website,
                "operational_name": c.operational_name,
            })
            company_blocks.append({
                "company_id": cid,
                "name": c.operational_name,
                "country": (c.address or {}).get("country"),
                "employee_count": c.employee_count,
                "revenue": c.revenue,
                "year_founded": c.year_founded,
                "is_public": c.is_public,
                "primary_naics": c.primary_naics,
                "business_model": c.business_model,
                "target_markets": c.target_markets,
                "core_offerings": c.core_offerings,
                "description": c.description,
            })

        prompt = JUDGE_BATCH_PROMPT.format(
            query=query,
            query_type=plan.get("query_type", "hybrid"),
            hypothetical_profile=plan.get("hypothetical_profile", ""),
            companies_json=json.dumps(company_blocks, ensure_ascii=False, default=str),
        )

        try:
            payload = self._generate_json(prompt, temperature=temperature)
        except ValueError:
            # Don't crash the batch -- mark everything in it as low-confidence
            # / unresolved so Stage 4 can still rank the rest of the run.
            return [
                {
                    "company_id": cb["company_id"],
                    "match": False,
                    "confidence": 0.0,
                    "reason": "local model failed to return parseable output for this batch",
                }
                for cb in company_blocks
            ]

        verdicts = payload.get("verdicts") if isinstance(payload, dict) else payload
        if not isinstance(verdicts, list):
            verdicts = []

        # Backfill any company the model silently dropped from its response,
        # so a partial answer never means a company disappears without a trace.
        returned_ids = {v.get("company_id") for v in verdicts if isinstance(v, dict)}
        for cb in company_blocks:
            if cb["company_id"] not in returned_ids:
                verdicts.append({
                    "company_id": cb["company_id"],
                    "match": False,
                    "confidence": 0.0,
                    "reason": "model omitted this company from its response",
                })
        return verdicts

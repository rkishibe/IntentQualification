import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from utils import company_id, expand_country_regions
from prompts import UNDERSTAND_QUERY_PROMPT, JUDGE_BATCH_PROMPT

class LocalLLMClient:
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

        plan.setdefault("query_type", "hybrid")
        plan.setdefault("structured_filters", {})
        plan.setdefault(
            "expansion_terms",
            [t for t in re.findall(r"[a-zA-Z]{4,}", query.lower())],
        )
        plan.setdefault(
            "hypothetical_profile",
            f"A company that satisfies: {query}",
        )

        # Normalize regional country filters
        plan["structured_filters"] = expand_country_regions(
            plan["structured_filters"]
        )

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

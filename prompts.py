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
                "description": "'structured': fully answered ONLY by explicit filters (location, size, revenue, public status). 'hybrid': has both a hard filter and a judgment component. 'semantic': needs judgment beyond raw fields (e.g. 'fast-growing', 'competing with traditional banks'). 'ecosystem': needs reasoning about a company's role in a supply chain or business relationship not stated directly (e.g. 'could supply packaging for a cosmetics brand')."
            }},
            "structured_filters": {{
                "type": "object",
                "description": "Only include keys the query actually constrains.",
                "properties": {{
                    "country": {{"type": "string"}}, //For geographic queries, if the user specifies one of these region groups, put the region name itself in the `country` field: Scandinavia, Nordic, Europe, Asia. Do not try to expand the region into individual countries.
                    "region": {{"type": "string"}},
                    "city": {{"type": "string"}},
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
                            "Use the MOST SPECIFIC NAICS prefix you can confidently identify for this industry. "
                            "Prefferably >=3. Only fall back to a 2-digit sector code if you genuinely cannot determine a more "
                            "specific one -- do not default to broad as a safe choice when the industry is "
                            "clearly identifiable (e.g. "automobile companies" -> use the automobile "
                            "manufacturing code, not the general manufacturing sector). "
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
    "Scandinavia": ["Sweden", "Norway", "Denmark"],
    "Nordics": ["Sweden", "Norway", "Denmark", "Finland", "Iceland"],
    "Europe": [
        "Austria", "Belgium", "Switzerland", "Germany", "Denmark", "Spain", "Finland", "France", "United Kingdom", "Greece", "Croatia",
        "Ireland", "Iceland", "Italy", "Latvia", "Luxembourg", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Sweden",
        "Ukraine",
    ],
    "Asia" : ["China", "Japan", "South Korea"]
}
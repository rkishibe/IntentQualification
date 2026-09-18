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
                            "11": "Agriculture, Forestry, Fishing and Hunting",
                            "21": "Mining, Quarrying, and Oil and Gas Extraction",
                            "22": "Utilities",
                            "23": "Construction",

                            "31": "Manufacturing",
                            "32": "Manufacturing",
                            "33": "Manufacturing",

                            "42": "Wholesale Trade",

                            "44": "Retail Trade",
                            "45": "Retail Trade",

                            "48": "Transportation and Warehousing",
                            "49": "Transportation and Warehousing",

                            "51": "Information",
                            "52": "Finance and Insurance",
                            "53": "Real Estate and Rental and Leasing",
                            "54": "Professional, Scientific, and Technical Services",
                            "55": "Management of Companies and Enterprises",
                            "56": "Administrative and Support and Waste Management and Remediation Services",
                            "61": "Educational Services",
                            "62": "Health Care and Social Assistance",
                            "71": "Arts, Entertainment, and Recreation",
                            "72": "Accommodation and Food Services",
                            "81": "Other Services (except Public Administration)",
                            "92": "Public Administration",
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
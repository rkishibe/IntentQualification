
  

# 3.1 Approach

  

## Describe your system architecture.

  

**What components does it include?**

  

Architecture has been fragmented into responsiblities and stages in the classificaition process as follows: data classes, data visualization, preprocessing, embedding client, LLM client. The orchestrator `QualificationSystem` plugs the four stages presented below and data flows through them in one direction as a funnel.

  

Before passing into the embedding and LLM stages, the jsonl file is preprocessed for data uniformity and saved into a DataFrame that is then embedded and rembedded based on changes.

  

**How do they interact?**

1.  `understand_query` (1 LLM call, cached per query) produces a structured plan based on key terms. The query can be "structured" aka it contains explicit filters such as lcoation, employee count, revenue or industry sector. Other option is "semantic" that implies the query uses terms that need semantic itnerpretation, such as 'fast-growing'. The "ecosystem" tag refers to a company's role within the supply chain, such as supplier, producer, intermediary. The last tag is "hybrid", which can contain both structured and interpretative filters. The function call returns a plan/profiling of the companies, which can contains the dterministic filters, semnatic terms and a description of the ideal company profile from the query.

2.  `structured_filter` (no LLM call) will apply the generated deterministic field constraints. The strucutred queries stop here and results are returned without additional LLM cost.

3.  `embedding_retrieval` scores companies against the raw query + hypothetical profile + expansion terms, using precomputed vectors loaded from embeddings.npz, keeping the top N=50 candidates.

4.  `batch_llm_jjudge` sends survivors to the LLM in batches, then combines structured/embedding/LLM signals into a single ranked score.

  

**Why did you choose this design?**

- Difficulty is proportional with computational cost. Structured queries take a few seconds to run locally.

- Each stage compensates for a different weakness and they trickle down from the more general filters (such as location) to very specific requests (such as "packaging suppliers for cosmetic brands"). It would have been very inefficient to run a very simple query through all stages. Additionally, through the stages, a certain portion of the query is analysed (from request type, to structured filters, to expansion terms and hypothetical profile)

- Separation of components and stages helped efficient debugging.

  
  
  
  

# 3.2 Tradeoffs

  
  

**What did you optimize for?**

- Cost: the system runs locally, using the embedding model `BAAI/bge-small-en-v1.5` with vectors precomputed once and cached to `embeddings.npz`, and a the `Qwen/Qwen2.5-7B-Instruct` model for query understanding and batch judging.

  

- Speed:

- query understanding runs once per query and is cached;

- structured queries short-circuit before the LLM stage entirely

- embeddings are precomputed, not recomputed per query

- judging is batched ~12 companies per call instead of one call per company

- self-consistency re-runs in `batch_llm_judge` are restricted to the borderline confidence limits (hardcoded) only to companies within that limit

  
  

**What trade-offs did you intentionally make?**

- Prompt granularity: complex instructions are harder to understand and NAICS codes have been abbreviated to the first two digits due to the model's limitations in computing the full codes.

  

- Recall over precision on missing data. After I visualized the data, I noticed some missing fields, especially in secondary NAICS, so those are flagged low-confidence, but not completely. The downside is that an unparseable NAICS code is not a rejection so over-inclusion is possible.

  

- Over-depending on `understand_query` for structured queries. I have noticed that regional locations or cities do not always return the correct filter, so structured queries might return nothing. I have tried to counteract this by manually parsing region names, such as Europe or Scandinavia, but it is not fully/efficiently covered.

  

- Borderline 0.4-06 limit. It is possible that some correct queries might be lying very close to this limit but are not included.

  

# 3.3 Error Analysis

  

**Where does your system struggle?**

  

Passing wrong companies with full confidence

  

- NAICS prefixes: Example Query: "Automobile companies in United States."The query-understanding model emits a 2 digit prefix ("33"Manufacturing) rather than 3361 startswith, this admitted Gillette, Candy, Drinks and Foods companies based on structured filtering. A prompt instruction is telling the LLM to choose the most precise code but the fall-back is 2 digits, which is what it usually what it outputs. Initially, the system struggled to output the correct first 2 digits of the NAICS code, but after refining the prompy, accuracy has improved. Yet, I believe the first 2 digits are not enough for an accurate result, so next steps would be to implement a soft-hard filtering system (I have started implementation, but due to time constraints, I have not managed to finish it)  
-  Country/location: If the country didn't match, the output would exclude companies from around the area. Also, `understand_query` struggled to understand world regions, but mapping the countries to continents fixed the problem.

- Small-model will struggle to output a result in a good amount of time and also can omit details in the query.

- Before upgrading the model, query understanding copied literal values out of the prompt's JSON schema example emitting `is_public: true` for queries that never mentioned public status, and `employee_count_min: 1000` / `year_founded_min: 2018` copied verbatim from the illustrative example.

- A hallucinated `is_public: true` excluded Dacia (`is_public: false`) from "Automobile companies in Romania" at stage 2.

- In batch judging, the same pattern produced verdicts reading "reason": "One short sentence" — the placeholder text from the prompt template. The drawback is that the model used now might run out of memory locally, garbage collecting is necessary.

  
  

# 3.4 Scaling

  

**If the system needed to handle 100,000 companies per query instead of 500, what would you change?**

  

- Retrieval: Stage 3 currently embeds candidates and computes similarity in a Python loop over the survivor set. Should be an approximate-nearest-neighbour index, so retrieval is sublinear rather than O(n) per query.

  

- Filter-before-retrieve ordering: Structured fields should move into an indexed db so the filter is a query.

  

- The live-embedding fallback becomes a liability: At 100k embedding, a cold cache will cause a computational failure

  

- Batch judging needs concurrency: currently sequential, so they must be parallelised or replaced with a hosted API

  
  

# 3.5 Failure Modes

  

**When might your system produce confident but incorrect results?**

  

- Structured queries have no safety net. This is where the example above on the very broad NAICS matches surfaced at score 1.0.

  

- Hallucinations: The fabricated JSON filters produce a shorter result list. False negatives are far harder to notice than false positives, because nothing in the output indicates the company was ever considered.

  

**What would you monitor in production to detect these failures?**

  

- Filter selectivity per stage: Log candidate counts entering and leaving each stage.

- Distribution of emitted NAICS prefix lengths: A spike in 1–2 digit prefixes signals the model losing precision

- Plan schema conformance: Validate every `understand_query` output against the key names, alert on unknown keys, type mismatches (str vs list in country, for example), or fields appearing that the query text doesn't support

- Rate of missing data pass-throughs: Track what fraction of results carry the low_confidence flag.

- Judge stage parse failure rate: Count verdicts returned as "model did not return a valid verdict" or backfilled defaults. A rising rate indicates truncation

- Placeholder leakage: Alert on verdict reasons or filter values that exactly match strings from the prompt templates (ex. "One short sentence").

- Score distribution by stage_reached: A large mass of results at exactly 1.0 from indicates queries short-circuiting to filter-only

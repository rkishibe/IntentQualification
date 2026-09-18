from __future__ import annotations

import os
import json
from typing import Optional
import pandas as pd
import numpy as np
import warnings
import gc
from data import Verdict
from utils import get_country, companies_from_dataframe, parse_nested, company_id, distance_km, geo_penalty, naics_filter_mode, _naics_codes
from embedding_client import LocalEmbeddingClient, build_embedding_dataset
from llm_client import LocalLLMClient, REGION_ALIASES


SEARCH_RADII_KM = [25, 50, 100, 200, 400]
MIN_GEO_RESULTS = 20

def structured_filter(companies: list, filters: dict) -> list:
    """Returns [(company, low_confidence), ...] for companies that pass."""
    out = []
    for c in companies:
        low_conf = False
        ok = True
        if ok and "country" in filters:
            country = (c.address or {}).get("country")
            if not country:
                low_conf = True
            else:
                targets = filters["country"]
                if isinstance(targets, str):
                    targets = [targets]
                if not any(
                    str(target).strip().lower() == country.strip().lower()
                    for target in targets
                ):
                    ok = False
        
        if ok and filters.get("employee_count_min") is not None:
            if c.employee_count is None:
                low_conf = True
            elif c.employee_count < filters["employee_count_min"]:
                ok = False

        if ok and filters.get("employee_count_max") is not None:
            if c.employee_count is None:
                low_conf = True
            elif c.employee_count > filters["employee_count_max"]:
                ok = False

        if ok and filters.get("revenue_min") is not None:
            if c.revenue is None:
                low_conf = True
            elif c.revenue < filters["revenue_min"]:
                ok = False

        if ok and filters.get("revenue_max") is not None:
            if c.revenue is None:
                low_conf = True
            elif c.revenue > filters["revenue_max"]:
                ok = False

        if ok and filters.get("year_founded_min") is not None:
            if c.year_founded is None:
                low_conf = True
            elif c.year_founded < filters["year_founded_min"]:
                ok = False

        if ok and filters.get("year_founded_max") is not None:
            if c.year_founded is None:
                low_conf = True
            elif c.year_founded > filters["year_founded_max"]:
                ok = False

        if ok and filters.get("is_public") is True:
            if c.is_public is None:
                mentions_public = any(
                    kw in (c.description or "").lower()
                    for kw in ["publicly traded", "listed on", "nyse", "nasdaq", "stock exchange"]
                )
                if mentions_public:
                    low_conf = True
                else:
                    ok = False
            elif c.is_public is not True:
                ok = False

        if ok and filters.get("naics_prefixes"):
            codes = _naics_codes(c)
            if not codes:
                warnings.warn(
                    f"No NAICS code for {c.operational_name!r} "
                    f"(raw: {c.primary_naics!r}) -- check normalize_naics.",
                    stacklevel=2,
                )
                ok = False
            else:
                matches = [
                    p for code in codes for p in filters["naics_prefixes"]
                    if code.startswith(p)
                ]
                strong_matches = [p for p in matches if len(p) >= 1]
                if not strong_matches:
                    ok = False

        if ok:
            out.append((c, low_conf))
    return out


def load_embedding_index(output_dir: str = "data/embeddings") -> dict:
    """company_id -> np.ndarray, loaded from the .npz written by build_embedding_dataset."""
    vectors_path = os.path.join(output_dir, "embeddings.npz")
    if not os.path.exists(vectors_path):
        return {}
    loaded = np.load(vectors_path, allow_pickle=False)
    return {
        str(cid): vec
        for cid, vec in zip(loaded["company_ids"], loaded["embeddings"])
    }


def embedding_retrieve(
    query: str,
    plan: dict,
    candidates: list,
    embedding_index: dict,
    embedding_client,
    top_n: int = 50,
) -> list:
    """Returns [(company, score), ...] sorted descending, length <= top_n."""
    if not candidates:
        return []

    query_texts = [query, plan.get("hypothetical_profile", "")] + plan.get("expansion_terms", [])[:8]
    query_texts = [t for t in query_texts if t]
    query_vecs = embedding_client.embed(query_texts)  # already normalized

    scored = []
    to_embed_live = []  # (index_in_scored_placeholder, company) needing on-the-fly embedding
    for c in candidates:
        cid = company_id({"website": c.website, "operational_name": c.operational_name})
        vec = embedding_index.get(cid)
        if vec is None:
            to_embed_live.append(c)
            continue
        sims = query_vecs @ vec
        score = 0.6 * sims.max() + 0.4 * sims.mean()
        scored.append((c, float(score)))

    if to_embed_live:
        live_vecs = embedding_client.embed([c.composite_text() for c in to_embed_live])
        for c, vec in zip(to_embed_live, live_vecs):
            sims = query_vecs @ vec
            score = 0.6 * sims.max() + 0.4 * sims.mean()
            scored.append((c, float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Stage 3 orchestration: batching + self-consistency on borderline cases
# ---------------------------------------------------------------------------

def _cid(c) -> str:
    return company_id({"website": c.website, "operational_name": c.operational_name})


def batched_llm_judge(
    query: str,
    plan: dict,
    candidates: list,
    llm_client,
    batch_size: int = 12,
    borderline_band: tuple = (0.4, 0.6),
    self_consistency_runs: int = 1,
) -> dict:
    by_id = {_cid(c): c for c, _ in candidates}
    companies = [c for c, _ in candidates]
    verdicts = {}

    for i in range(0, len(companies), batch_size):
        batch = companies[i:i + batch_size]
        for r in llm_client.judge_batch(query, plan, batch):
            verdicts[r["company_id"]] = r

    borderline_ids = [
        cid for cid, v in verdicts.items()
        if borderline_band[0] <= v["confidence"] <= borderline_band[1]
    ]
    print("borderline", len(borderline_ids))
    for cid in borderline_ids:
        c = by_id[cid]
        votes = []
        for _ in range(self_consistency_runs):
            run = llm_client.judge_batch(query, plan, [c], temperature=0.7)
            votes.append(run[0])
        match_votes = [v["match"] for v in votes]
        majority_match = sum(match_votes) > len(match_votes) / 2
        avg_confidence = sum(v["confidence"] for v in votes) / len(votes)
        agreement = sum(1 for v in match_votes if v == majority_match)
        verdicts[cid] = {
            "company_id": cid,
            "match": majority_match,
            "confidence": round(avg_confidence, 2),
            "reason": f"self-consistency vote ({agreement}/{len(votes)} runs agreed): " + votes[0]["reason"],
        }

    return verdicts


# ---------------------------------------------------------------------------
# Stage 4: fusion & ranking
# ---------------------------------------------------------------------------

def fuse_and_rank(
    query_type: str,
    structured_pass: list,
    embedding_scores: dict,
    llm_verdicts: dict,
    geo_distances=None,
) -> list:
    geo_distances = geo_distances or {}

    results = []

    for c, low_conf in structured_pass:
        cid = _cid(c)

        e = embedding_scores.get(cid)
        v = llm_verdicts.get(cid)

        # ---------------------------------------------------------------
        # Structured-only query
        # ---------------------------------------------------------------
        if query_type == "structured" and e is None and v is None:
            score = 1.0 - (0.15 if low_conf else 0.0)

            # Geographic penalty
            distance = geo_distances.get(cid)
            score -= geo_penalty(distance)
            score = max(0.0, score)

            results.append(
                Verdict(
                    c,
                    score,
                    True,
                    "matched all structured filters",
                    "filter",
                )
            )
            continue

        # ---------------------------------------------------------------
        # Semantic queries: must have been retrieved by embeddings
        # ---------------------------------------------------------------
        if query_type != "structured" and e is None:
            continue

        # ---------------------------------------------------------------
        # LLM rejection
        # ---------------------------------------------------------------
        if v is not None and not v["match"]:
            continue

        # ---------------------------------------------------------------
        # Base score
        # ---------------------------------------------------------------
        w_struct, w_embed, w_llm = 0.2, 0.3, 0.5

        parts = []
        weights = []

        parts.append(
            1.0 - (0.15 if low_conf else 0.0)
        )
        weights.append(w_struct)

        if e is not None:
            parts.append(max(0.0, min(1.0, e)))
            weights.append(w_embed)

        if v is not None:
            parts.append(v["confidence"])
            weights.append(w_llm)

        score = sum(
            p * w
            for p, w in zip(parts, weights)
        ) / sum(weights)

        # ---------------------------------------------------------------
        # Geographic penalty
        # ---------------------------------------------------------------
        distance = geo_distances.get(cid)
        score -= geo_penalty(distance)
        score = max(0.0, score)

        # ---------------------------------------------------------------
        # Result metadata
        # ---------------------------------------------------------------
        reason = (
            v["reason"]
            if v
            else (
                "passed structured filters, ranked by semantic similarity"
                if e is not None
                else "matched structured filters"
            )
        )

        stage = (
            "llm"
            if v is not None
            else (
                "embedding"
                if e is not None
                else "filter"
            )
        )

        results.append(
            Verdict(
                c,
                score,
                True,
                reason,
                stage,
            )
        )

    results.sort(
        key=lambda r: r.score,
        reverse=True,
    )

    return results


class QualificationSystem:
    def __init__(
        self,
        llm_client,
        embedding_client,
        embedding_index: Optional[dict] = None,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.embedding_index = embedding_index or {}
        self._query_plan_cache = {}

    def _get_plan(self, query: str) -> dict:
        key = query.strip().lower()

        if key not in self._query_plan_cache:
            self._query_plan_cache[key] = (
                self.llm_client.understand_query(query)
            )

        return self._query_plan_cache[key]

    def _geo_search(
        self,
        companies: list,
        plan: dict,
        min_results: int = MIN_GEO_RESULTS,
    ) -> list:
        """
        Search companies around the geographic location specified in the plan.

        Returns:
            [(company, distance_km), ...]

        The search starts with the smallest radius and expands until
        min_results are found, or until the largest radius is reached.
        """

        filters = plan.get("structured_filters", {})

        latitude = filters.get("latitude")
        longitude = filters.get("longitude")

        # No geographic constraint.
        if latitude is None or longitude is None:
            return [(company, None) for company in companies]

        last_results = []

        for radius_km in SEARCH_RADII_KM:
            results = []

            for company in companies:
                address = company.address or {}

                lat = address.get("latitude")
                lon = address.get("longitude")

                if lat is None or lon is None:
                    continue

                distance = distance_km(
                    float(latitude),
                    float(longitude),
                    float(lat),
                    float(lon),
                )

                if distance <= radius_km:
                    results.append((company, distance))

            if len(results) >= min_results:
                return results

            last_results = results

        return last_results

    def run(
            self,
            query: str,
            companies: list,
            top_n_embed: int = 40,
        ) -> list:
            print("START RUN")
            # ------------------------------------------------------------------
            # 1. Understand query
            # ------------------------------------------------------------------

            plan = self._get_plan(query)
            qtype = plan["query_type"]

            # Defensive net
            if qtype in ("ecosystem", "semantic"):
                plan["structured_filters"].pop("naics_prefixes", None)

            # ------------------------------------------------------------------
            # 2. Hard structured filters
            # ------------------------------------------------------------------

            mode = naics_filter_mode(companies, plan["structured_filters"])
            print(mode)

            structured_pass = structured_filter(
                companies,
                plan["structured_filters"],
            )

            print("Structured pass:", structured_pass)

            candidates = [c for c, _ in structured_pass]

            # ------------------------------------------------------------------
            # 3. Geographic search
            # ------------------------------------------------------------------

            geo_candidates = self._geo_search(
                candidates,
                plan,
            )

            print("Geo candidates:", geo_candidates)

            candidates = [
                company
                for company, _distance in geo_candidates
            ]

            geo_distances = {
                _cid(company): distance
                for company, distance in geo_candidates
            }

            # ------------------------------------------------------------------
            # 4. Structured-only query
            # ------------------------------------------------------------------

            if qtype == "structured":
                return fuse_and_rank(
                    qtype,
                    structured_pass,
                    {},
                    {},
                    geo_distances=geo_distances,
                )

            # ------------------------------------------------------------------
            # 5. Semantic retrieval
            # ------------------------------------------------------------------

            embed_ranked = embedding_retrieve(
                query,
                plan,
                candidates,
                self.embedding_index,
                self.embedding_client,
                top_n=top_n_embed,
            )

            print("Embed_ranked:", embed_ranked)

            embedding_scores = {
                _cid(c): s
                for c, s in embed_ranked
            }

            # ------------------------------------------------------------------
            # 6. Hybrid shortcut
            # ------------------------------------------------------------------

            if qtype == "hybrid" and len(embed_ranked) <= 15:
                return fuse_and_rank(
                    qtype,
                    structured_pass,
                    embedding_scores,
                    {},
                    geo_distances=geo_distances,
                )



            # ------------------------------------------------------------------
            # 7. LLM judging
            # ------------------------------------------------------------------

            llm_verdicts = batched_llm_judge(
                query,
                plan,
                embed_ranked,
                self.llm_client,
            )

            print("LLM Verdicts:", llm_verdicts)

            # ------------------------------------------------------------------
            # 8. Final ranking
            # ------------------------------------------------------------------

            return fuse_and_rank(
                qtype,
                structured_pass,
                embedding_scores,
                llm_verdicts,
                geo_distances=geo_distances,
            )


if __name__=="__main__":

    df = pd.read_json("data/companies.jsonl", lines=True)

    embedding_client = LocalEmbeddingClient(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
    )

    companies = companies_from_dataframe(df)

    build_embedding_dataset(
        companies,
        embedding_client,
        output_dir="data/embeddings",
    )

    llm_client = LocalLLMClient()

    embedding_index = load_embedding_index("data/embeddings")
    print(f"Loaded {len(embedding_index)} cached embeddings")
    
    print("instantiate qualitification system")
    
    system = QualificationSystem(llm_client, embedding_client, embedding_index=embedding_index)
    
    # plan = llm_client.understand_query("B2B SaaS companies providing HR solutions in Europe")
    # print(json.dumps(plan, indent=2))

    gc.collect()
    #raise SystemExit
    example_queries = [
    "Turbine manufacturers in Europe.",
    #"Public software companies with more than 1,000 employees.",
    # "Food and beverage manufacturers in France",
    #"Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    # "Construction companies in the United States with revenue over $50 million",
    # "Pharmaceutical companies in Switzerland",
    #"B2B SaaS companies providing HR solutions in Europe",
    # "Clean energy startups founded after 2018 with fewer than 200 employees",
    #"Fast-growing fintech companies competing with traditional banks in Europe.",
    #"E-commerce companies using Shopify or similar platforms",
    #"Renewable energy equipment manufacturers in Scandinavia",
    #"Companies that manufacture or supply critical components for electric vehicle battery production",
    ]

    for q in example_queries:
        print(f"\n=== {q!r} ===")
        results = system.run(q, companies)
        print(f"  ({len(results)} total matches)")
        for r in results:
            print(f"  [{r.stage_reached:9s}] {r.score:.2f}  {r.company.operational_name:30s} - {r.reason}")


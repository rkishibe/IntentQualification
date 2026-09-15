from __future__ import annotations

import os
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional
import ast
import pandas as pd
import pycountry
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
from pathlib import Path
from data import Company, Verdict

def normalize_nested(value): # address is both a JSON object and a Python repr string, this function normalizes the data
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        raise TypeError(
            f"Expected dict/string/None, got {type(value).__name__}"
        )

    value = value.strip()

    if not value:
        return None

    # JSON
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Python representation
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Could not parse nested object: {value!r}"
            ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Parsed value is not an object: {parsed!r}"
        )

    return parsed

def parse_nested(value):
    if pd.isna(value):
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)

    return None

def get_country(address):
    address = parse_nested(address)

    if not address:
        return None

    # Already-normalized country name
    if address.get("country"):
        return address["country"]

    # ISO country code
    code = address.get("country_code")

    if code:
        country = pycountry.countries.get(
            alpha_2=str(code).upper()
        )

        if country:
            return country.name

        return str(code).upper()

    return None

# ---------------------------------------------------------------------------
# Loading companies from the dataset
#
#     df = pd.read_json("data/companies.jsonl", lines=True)
#     companies = companies_from_dataframe(df)
#
# Robust to missing columns, NaN cells (pandas' native representation of a
# missing JSON field), and fields that arrive as plain strings instead of
# lists/dicts -- real company datasets are inconsistent about all of this.
# ---------------------------------------------------------------------------

def _is_na(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    return False


def _as_str(val: Any, default: str = "") -> str:
    return default if _is_na(val) else str(val)


def _as_int(val: Any) -> Optional[int]:
    if _is_na(val):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def _as_float(val: Any) -> Optional[float]:
    if _is_na(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _as_bool(val: Any) -> Optional[bool]:
    if _is_na(val):
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1", "public", "y"}
    return bool(val)


def _as_dict(val: Any) -> Optional[dict]:
    """For primary_naics: {"code": ..., "label": ...}."""
    if _is_na(val):
        return None
    if isinstance(val, dict):
        return val
    # pandas sometimes leaves nested JSON as a numpy record/other mapping-like
    try:
        return dict(val)
    except (TypeError, ValueError):
        return None

def normalize_naics(naics): 

    if naics is None:
        return None

    # Already a dict
    if isinstance(naics, dict):
        naics = dict(naics)

    # String representation
    elif isinstance(naics, str):
        naics = naics.strip()

        if not naics:
            return None

        # Try valid JSON first
        try:
            naics = json.loads(naics)
        except json.JSONDecodeError:
            # Fall back to Python repr
            naics = ast.literal_eval(naics)

        if not isinstance(naics, dict):
            raise ValueError(
                f"Expected naics to parse into a dict, got {type(naics).__name__}"
            )

    else:
        raise TypeError(
            f"Unexpected naics type: {type(naics).__name__}"
        )

    return naics

def normalize_address(address): #normalize ISO country code into full country name and parse addr into a dict

    if address is None:
        return None

    # Already a dict
    if isinstance(address, dict):
        address = dict(address)

    # String representation
    elif isinstance(address, str):
        address = address.strip()

        if not address:
            return None

        # Try valid JSON first
        try:
            address = json.loads(address)
        except json.JSONDecodeError:
            # Fall back to Python repr
            address = ast.literal_eval(address)

        if not isinstance(address, dict):
            raise ValueError(
                f"Expected address to parse into a dict, got {type(address).__name__}"
            )

    else:
        raise TypeError(
            f"Unexpected address type: {type(address).__name__}"
        )

    # Normalize country
    country_code = address.pop("country_code", None)

    if country_code:
        country = pycountry.countries.get(
            alpha_2=str(country_code).upper()
        )

        if country:
            address["country"] = country.name
        else:
            # Preserve unexpected/unknown codes
            address["country"] = country_code

    return address


def _as_list(val: Any) -> list:
    """For business_model / core_offerings / target_markets."""
    if _is_na(val):
        return []
    if isinstance(val, (list, tuple)):
        return [v for v in val if not _is_na(v)]
    if isinstance(val, str):
        # tolerate a single stringly-typed value or a comma-separated one
        return [s.strip() for s in val.split(",") if s.strip()]
    return []

def company_from_row(row: dict) -> Company:
    return Company(
        operational_name=_as_str(row.get("operational_name")),
        website=_as_str(row.get("website")),
        year_founded=_as_int(row.get("year_founded")),
        address=normalize_address(row.get("address")),
        employee_count=_as_int(row.get("employee_count")),
        revenue=_as_float(row.get("revenue")),
        primary_naics=normalize_naics(row.get("primary_naics")),
        secondary_naics=normalize_naics(row.get("secondary_naics")),
        description=_as_str(row.get("description")),
        business_model=_as_list(row.get("business_model")),
        core_offerings=_as_list(row.get("core_offerings")),
        target_markets=_as_list(row.get("target_markets")),
        is_public=_as_bool(row.get("is_public")),
    )


def companies_from_dataframe(df: pd.DataFrame) -> list[Company]:
    """
    Convert a DataFrame of company records -- e.g.
    `pd.read_json("data/companies.jsonl", lines=True)` -- into a list of
    `Company` objects. Missing columns/NaN cells all degrade gracefully to
    the same "unknown field" state the rest of the pipeline already expects
    (see Stage 1's missing-data policy).
    """
    companies = []
    for _, row in df.iterrows():
        companies.append(company_from_row(row.to_dict()))
    return companies

def company_id(row: dict) -> str:
    """
    Generate a deterministic ID for a company.
    """
    identity = "|".join([
        str(row.get("website") or "").strip().lower(),
        str(row.get("operational_name") or "").strip().lower(),
    ])

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

def text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
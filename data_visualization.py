import pandas as pd
from utils import normalize_address, get_country, parse_nested

def country_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with the distribution of companies by country.
    """
    # Normalize the address column to extract country names
    df['normalized_address'] = df['address'].apply(normalize_address)
    df['country'] = df['normalized_address'].apply(lambda x: x.get('country') if isinstance(x, dict) else None)

    # Count the number of companies per country
    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['country', 'count']

    return country_counts

def naics_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with the distribution of companies by primary NAICS code.
    """
    # Normalize the primary_naics column to extract NAICS codes
    df['normalized_primary_naics'] = df['primary_naics'].apply(lambda x: x.get('code') if isinstance(x, dict) else None)

    # Count the number of companies per NAICS code
    naics_counts = df['normalized_primary_naics'].value_counts().reset_index()
    naics_counts.columns = ['naics_code', 'count']

    return naics_counts

if __name__ == "__main__":
    df = pd.read_json("data/companies.jsonl", lines=True)
    null_naics = df["primary_naics"].isna().sum()
    print(f"Null primary_naics values: {null_naics}")

    null_naics = df["secondary_naics"].isna().sum()
    print(f"Null secondary_naics values: {null_naics}")

    completeness = pd.DataFrame({
    "field": df.columns,
    "missing": df.isna().sum().values,
    "missing_pct": (df.isna().mean() * 100).round(2).values,
    "present": df.notna().sum().values,
    "present_pct": (df.notna().mean() * 100).round(2).values,
    })

    completeness = completeness.sort_values(
        "missing_pct",
        ascending=False
    ).reset_index(drop=True)

    print(completeness.to_string(index=False))

    # ------------------------------------------------------------
    # 1. PRIMARY NAICS DISTRIBUTION
    # ------------------------------------------------------------

    def get_naics(value):
        value = parse_nested(value)

        if not value:
            return None

        return value.get("code")


    def get_naics_label(value):
        value = parse_nested(value)

        if not value:
            return None

        return value.get("label")


    naics = pd.DataFrame({
        "code": df["primary_naics"].apply(get_naics),
        "label": df["primary_naics"].apply(get_naics_label),
    })

    naics_distribution = (
        naics
        .value_counts(["code", "label"], dropna=False)
        .reset_index(name="count")
    )

    naics_distribution["pct"] = (
        naics_distribution["count"] / len(df) * 100
    ).round(2)

    print("\nPRIMARY NAICS DISTRIBUTION")
    print(naics_distribution.to_string(index=False))



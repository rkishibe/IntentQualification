import pandas as pd
from utils import normalize_address

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




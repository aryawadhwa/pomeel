"""
Data Loader Module for World Bank Global Findex 2021 India Microdata.
STRICT REQUIREMENT: NO SYNTHETIC VALUES, NO DUMMY DATA.
Loads authentic, official microdata files exclusively.
"""

import os
from typing import Optional
import pandas as pd


OFFICIAL_PORTAL_URL = "https://microdata.worldbank.org/index.php/catalog/4651"
OFFICIAL_STUDY_ID = "IND_2021_FINDEX_v02_M"


def load_findex_data(data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Loads authentic, official World Bank Global Findex 2021 India microdata.
    
    Checks for the official World Bank microdata files:
      - IND_2021_FINDEX.dta (Stata format)
      - IND_2021_FINDEX.csv
      - micro_world.dta (Global microdata filtered to economycode == 'IND')
      - micro_world.csv (Global microdata filtered to economycode == 'IND')
      
    If no official microdata file is found, raises FileNotFoundError with explicit
    instructions on how to obtain the official file from the World Bank repository.
    NO DUMMY OR SYNTHETIC DATA IS PERMITTED.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory holding raw microdata.
        
    Returns
    -------
    pd.DataFrame
        Authentic World Bank survey microdata for India.
        
    Raises
    ------
    FileNotFoundError
        If no authentic microdata file is found in data_dir.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    candidate_files = [
        'IND_2021_FINDEX.dta',
        'IND_2021_FINDEX.csv',
        'micro_world_india.csv',
        'micro_world.dta',
        'micro_world.csv'
    ]
    
    for filename in candidate_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"[AUTHENTIC DATA] Loading official World Bank microdata from: {filepath}")
            if filename.endswith('.dta'):
                df = pd.read_stata(filepath)
            else:
                df = pd.read_csv(filepath, low_memory=False)
                
            # Filter to India if global dataset
            if 'economycode' in df.columns:
                ind_mask = df['economycode'].str.upper() == 'IND'
                if ind_mask.any():
                    df = df[ind_mask].copy()
                    print(f"[AUTHENTIC DATA] Filtered global Findex to India sub-sample (N={len(df):,}).")
            elif 'economy' in df.columns:
                ind_mask = df['economy'].str.lower() == 'india'
                if ind_mask.any():
                    df = df[ind_mask].copy()
                    print(f"[AUTHENTIC DATA] Filtered global Findex to India sub-sample (N={len(df):,}).")
                    
            print(f"[AUTHENTIC DATA] Successfully loaded {len(df):,} authentic survey records.")
            return df

    error_msg = (
        f"\n{'='*75}\n"
        f"STRICT PROTOCOL ENFORCEMENT: NO DUMMY OR SYNTHETIC DATA PERMITTED.\n"
        f"{'='*75}\n"
        f"No authentic World Bank Global Findex microdata file was detected in:\n"
        f"  '{os.path.abspath(data_dir)}/'\n\n"
        f"To execute the empirical pipeline on official data:\n"
        f"1. Navigate to the official World Bank Microdata Library portal:\n"
        f"   URL: {OFFICIAL_PORTAL_URL}\n"
        f"   Study Reference ID: {OFFICIAL_STUDY_ID}\n"
        f"2. Sign in with a World Bank account and download the microdata file:\n"
        f"   - 'IND_2021_FINDEX.dta' OR 'IND_2021_FINDEX.csv'\n"
        f"   (or the global 'micro_world.dta' / 'micro_world.csv')\n"
        f"3. Place the downloaded file into '{data_dir}/'\n"
        f"4. Re-run this script.\n"
        f"{'='*75}"
    )
    raise FileNotFoundError(error_msg)


if __name__ == '__main__':
    try:
        df = load_findex_data()
        print(f"Data columns: {list(df.columns[:10])}...")
    except FileNotFoundError as e:
        print(e)

"""
Data Loader Module for World Bank Global Findex 2021 India Microdata.
STRICT REQUIREMENT: NO SYNTHETIC VALUES, NO DUMMY DATA.
Loads authentic, official microdata files exclusively.
"""

import os
from typing import Optional
import pandas as pd


OFFICIAL_PORTAL_URL = "https://microdata.worldbank.org/index.php/catalog/4653"
OFFICIAL_STUDY_ID = "IND_2021_FINDEX_v02_M"


def load_findex_data(data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Loads authentic, official World Bank Global Findex microdata for India.
    Supports both 2021 wave (micro_ind.csv) and 2024 wave (Findex_Microdata_2025_updateIndia.csv).
    Combines them into a multi-wave longitudinal panel with survey_wave tags.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    loaded_dfs = []
    
    # Check for 2021 wave
    f21_candidates = ['micro_ind.csv', 'IND_2021_FINDEX.csv', 'IND_2021_FINDEX.dta']
    for f in f21_candidates:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            print(f"[AUTHENTIC DATA] Loading 2021 Findex microdata: {p}")
            df21 = pd.read_csv(p, low_memory=False) if f.endswith('.csv') else pd.read_stata(p)
            df21 = df21.assign(survey_wave=2021)
            loaded_dfs.append(df21)
            print(f"  -> Successfully loaded 2021 wave (N={len(df21):,}).")
            break
            
    # Check for 2024 wave
    f24_candidates = [
        'Findex_Microdata_2025_updateIndia.csv',
        'IND_2024_FINDEX.csv',
        'IND_2024_FINDEX.dta'
    ]
    for f in f24_candidates:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            print(f"[AUTHENTIC DATA] Loading 2024 Findex microdata: {p}")
            df24 = pd.read_csv(p, low_memory=False) if f.endswith('.csv') else pd.read_stata(p)
            df24 = df24.assign(survey_wave=2024)
            loaded_dfs.append(df24)
            print(f"  -> Successfully loaded 2024 wave (N={len(df24):,}).")
            break
            
    # Fallback to any other .dta or .csv if the above named files weren't found
    if not loaded_dfs:
        for f in os.listdir(data_dir):
            if (f.endswith('.csv') or f.endswith('.dta')) and not f.startswith('.'):
                p = os.path.join(data_dir, f)
                print(f"[AUTHENTIC DATA] Loading microdata file: {p}")
                df_other = pd.read_csv(p, low_memory=False) if f.endswith('.csv') else pd.read_stata(p)
                if 'year' in df_other.columns:
                    df_other['survey_wave'] = df_other['year']
                else:
                    df_other['survey_wave'] = 2021
                loaded_dfs.append(df_other)

    if not loaded_dfs:
        raise FileNotFoundError(
            f"No authentic World Bank microdata files found in '{data_dir}/'."
        )
        
    combined_df = pd.concat(loaded_dfs, ignore_index=True)
    print(f"[AUTHENTIC DATA] Master empirical dataset ready: {len(combined_df):,} records across {len(loaded_dfs)} wave(s).")
    return combined_df

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

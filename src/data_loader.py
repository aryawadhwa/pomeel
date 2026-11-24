"""
Data Loader Module for World Bank Global Findex Microdata (India).
STRICT REQUIREMENT: NO SYNTHETIC VALUES, NO DUMMY DATA.
Loads authentic, official microdata files exclusively.
Supports both Global Findex 2021 (IND_2021_FINDEX) and the newly released
Global Findex 2025 / 2024 wave (IND_2024_FINDEX), pooling them into a
multi-wave longitudinal panel when both are present.
"""

import os
import pandas as pd


OFFICIAL_PORTAL_URL_2021 = "https://microdata.worldbank.org/index.php/catalog/4653"
OFFICIAL_STUDY_ID_2021 = "IND_2021_FINDEX_v02_M"

OFFICIAL_PORTAL_URL_2025 = "https://microdata.worldbank.org/index.php/catalog/7916"
OFFICIAL_STUDY_ID_2025 = "IND_2024_FINDEX_v02_M"


def load_findex_data(data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Loads authentic, official World Bank Global Findex microdata for India.
    Supports both 2021 wave (micro_ind.csv / IND_2021_FINDEX) and 2024/2025 wave
    (Findex_Microdata_2025_updateIndia.csv / IND_2024_FINDEX), combining them
    into a multi-wave longitudinal panel with survey_wave tags.
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

    # Check for 2024 / 2025 wave
    f24_candidates = [
        'Findex_Microdata_2025_updateIndia.csv',
        'IND_2024_FINDEX.csv',
        'IND_2024_FINDEX.dta',
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

    # Fallback to any other .dta or .csv (e.g., global micro_world filtered to IND)
    if not loaded_dfs:
        for f in os.listdir(data_dir):
            if (f.endswith('.csv') or f.endswith('.dta')) and not f.startswith('.'):
                p = os.path.join(data_dir, f)
                print(f"[AUTHENTIC DATA] Loading microdata file: {p}")
                df_other = pd.read_csv(p, low_memory=False) if f.endswith('.csv') else pd.read_stata(p)
                if 'economycode' in df_other.columns:
                    ind_mask = df_other['economycode'].astype(str).str.upper() == 'IND'
                    if ind_mask.any():
                        df_other = df_other[ind_mask].copy()
                if 'year' in df_other.columns:
                    df_other = df_other.assign(survey_wave=df_other['year'])
                elif '2024' in f or '2025' in f:
                    df_other = df_other.assign(survey_wave=2024)
                else:
                    df_other = df_other.assign(survey_wave=2021)
                loaded_dfs.append(df_other)

    if loaded_dfs:
        combined_df = pd.concat(loaded_dfs, ignore_index=True)
        print(
            f"[AUTHENTIC DATA] Master empirical dataset ready: {len(combined_df):,} records "
            f"across {len(loaded_dfs)} wave(s)."
        )
        return combined_df

    error_msg = (
        f"\n{'='*75}\n"
        f"STRICT PROTOCOL ENFORCEMENT: NO DUMMY OR SYNTHETIC DATA PERMITTED.\n"
        f"{'='*75}\n"
        f"No authentic World Bank Global Findex microdata file was detected in:\n"
        f"  '{os.path.abspath(data_dir)}/'\n\n"
        f"To execute the empirical pipeline on official data, obtain either:\n"
        f"A) Global Findex 2025 Edition (2024 survey wave with Digital Connectivity Tracker):\n"
        f"   - URL: {OFFICIAL_PORTAL_URL_2025}\n"
        f"   - Study Reference ID: {OFFICIAL_STUDY_ID_2025}\n"
        f"   - File name: 'Findex_Microdata_2025_updateIndia.csv' or 'IND_2024_FINDEX.csv'\n"
        f"B) Global Findex 2021 Benchmark Wave:\n"
        f"   - URL: {OFFICIAL_PORTAL_URL_2021}\n"
        f"   - Study Reference ID: {OFFICIAL_STUDY_ID_2021}\n"
        f"   - File name: 'micro_ind.csv' or 'IND_2021_FINDEX.csv'\n\n"
        f"Place the downloaded file into '{data_dir}/' and re-run.\n"
        f"{'='*75}"
    )
    raise FileNotFoundError(error_msg)


if __name__ == '__main__':
    try:
        df = load_findex_data()
        print(f"Data columns: {list(df.columns[:10])}...")
    except FileNotFoundError as e:
        print(e)

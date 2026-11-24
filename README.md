# Capable-but-Cash: Explainable ML on India's Currency Demand Paradox

[![World Bank Findex](https://img.shields.io/badge/Data-World%20Bank%20Global%20Findex%202021-blue.svg)](https://microdata.worldbank.org/index.php/catalog/4653)
[![RBI Statistics](https://img.shields.io/badge/Monetary%20Data-Reserve%20Bank%20of%20India-green.svg)](https://www.rbi.org.in/)
[![NPCI Statistics](https://img.shields.io/badge/Retail%20Payments-NPCI%20UPI-orange.svg)](https://www.npci.org.in/)

**Research Project:** *Analyzing India’s Rising Currency in Circulation: An Explainable Machine Learning Approach to the "Capable-but-Cash" Paradox*  
**Authors:** Aarush Garg, Arya Wadhwa, Dilraj Singh  
**Institution:** RV College of Engineering, Bengaluru, India  

---

## 📌 Strict Empirical Research Governance
* **Zero Dummy or Synthetic Data:** All empirical claims, values, and baseline figures are directly cited from official institutional publications (World Bank, Reserve Bank of India, National Payments Corporation of India).
* **Direct Citation Standard:** Every single theoretical term (e.g., UTAUT Effort Expectancy, Status-Quo Bias, Pain of Paying, The Cash-Out Loop) and empirical metric is cited with exact scholarly and institutional attribution.

---

## 📊 Directly Cited Empirical Baseline

| Metric | Empirical Value | Primary Scholarly / Institutional Source |
| :--- | :---: | :--- |
| **India Absolute Currency in Circulation (CiC)** | **₹42.8 Lakh Crore** | Reserve Bank of India (RBI, 2024) *Weekly Statistical Supplement* |
| **Annualized CiC Growth Rate** | **11.2%** | Reserve Bank of India (RBI, 2024) *Money Stock Measures* |
| **Monthly UPI Transaction Volume** | **14.96 Billion** | NPCI (2024) *UPI Product Statistics* (October 2024) |
| **Monthly UPI Transaction Value** | **₹20.64 Lakh Crore** | NPCI (2024) *UPI Product Statistics* (October 2024) |
| **Adult Account Ownership in India** | **77.5% ($\approx$78%)** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 25) |
| **Share of Inactive Accounts in India** | **35.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 32) |
| **Banked Adults Paying Merchants Only in Cash** | **670 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Banked Women Paying Merchants Only in Cash** | **350 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Adult Mobile Phone Ownership** | **72.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 28) |
| **Adults Making Digital Merchant Payments** | **12.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 56) |
| **Wage Earners Receiving Wages in Cash** | **62.4%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 64) |
| **Stated Barrier: Habit / "Used to it"** | **78.0%** | Global Findex 2021 Survey / Project Baseline Survey |
| **Stated Barrier: Merchant Accepts Cash Only** | **10.0%** | Global Findex 2021 Survey / Project Baseline Survey |
| **Stated Barrier: Distrust & Scam Anxiety** | **4.9%** | Global Findex 2021 Survey / Project Baseline Survey |
| **Stated Barrier: Higher Digital Costs** | **2.1%** | Global Findex 2021 Survey / Project Baseline Survey |

---

## 📂 Project Architecture

```
.
├── Capable-but-Cash_Experiential_Learning_Phase_I.pdf   # Phase I Project Presentation
├── LITERATURE_REVIEW.md                                 # Systematic literature review (33 peer-reviewed citations)
├── README.md                                           # Project governance & execution guide
├── data/
│   ├── raw/                                            # Raw World Bank Global Findex microdata (.dta / .csv)
│   └── processed/                                      # Filtered "Capable" cohort microdata
├── figures/                                            # Model performance and interpretability outputs
├── paper/
│   ├── RESEARCH_PAPER.md                               # Complete research paper manuscript (100% cited)
│   └── paper.tex                                       # IEEE-format conference manuscript for Overleaf
└── src/
    ├── data_loader.py                                  # Strict authentic Findex loader (fails on missing official file)
    ├── preprocessor.py                                 # Cohort screening (Account & Phone & Internet)
    ├── models.py                                       # Survey-weighted Logit & regularized XGBoost
    ├── explainability.py                               # TreeSHAP game-theoretic interpretability engine
    └── main.py                                         # Master pipeline orchestrator
```

---

## 🚀 Running the Empirical Pipeline on Official Data

Due to World Bank privacy terms, official survey microdata must be downloaded directly by the researcher:
1. Visit the [World Bank Microdata Library - India: Global Findex 2021](https://microdata.worldbank.org/index.php/catalog/4653) (Study Reference ID: `IND_2021_FINDEX_v02_M`).
2. Download `IND_2021_FINDEX.dta` or `IND_2021_FINDEX.csv` (or the global `micro_world.dta` / `micro_world.csv`).
3. Place the file into `data/raw/`.
4. Run the master pipeline:
   ```bash
   python3 src/main.py
   ```
   *The data loader strictly verifies authentic files and prevents any synthetic execution.*

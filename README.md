# Capable-but-Cash: Explainable ML on India's Currency Demand Paradox

[![World Bank Findex](https://img.shields.io/badge/Data-World%20Bank%20Global%20Findex%20(2021%20%7C%202025)-blue.svg)](https://microdata.worldbank.org/index.php/catalog/7916)
[![RBI Statistics](https://img.shields.io/badge/Monetary%20Data-Reserve%20Bank%20of%20India-green.svg)](https://www.rbi.org.in/)
[![NPCI Statistics](https://img.shields.io/badge/Retail%20Payments-NPCI%20UPI-orange.svg)](https://www.npci.org.in/)

**Research Project:** *Analyzing India’s Rising Currency in Circulation: An Explainable Machine Learning Approach to the "Capable-but-Cash" Paradox*  
**Authors:** Aarush Garg, Arya Wadhwa, Dilraj Singh  
**Institution:** RV College of Engineering, Bengaluru, India  

---

## 📌 Strict Empirical Research Governance
* **Zero Dummy or Synthetic Data:** All empirical claims, values, and baseline figures are directly cited from official institutional publications (World Bank, Reserve Bank of India, National Payments Corporation of India).
* **Direct Citation Standard:** Every single theoretical term (e.g., UTAUT Effort Expectancy, Status-Quo Bias, Pain of Paying, The Cash-Out Loop) and empirical metric is cited with exact scholarly and institutional attribution.
* **Dual Market Friction Modeling:** Investigates both **consumer behavioral inertia** (habit, cognitive switching costs) and **merchant-side tax compliance friction** (AIS/TIS automated tracking and Section 63 GST scrutiny).

---

## 📊 Directly Cited Empirical Baseline

| Metric | Empirical Value | Primary Scholarly / Institutional Source |
| :--- | :---: | :--- |
| **India Banknotes in Circulation (Mar 2026)** | **₹41.24 Lakh Crore** | Reserve Bank of India (RBI, 2026) *Annual Report 2025–26* (Total CiC ₹41.7–43.5L Cr) |
| **Annualized Banknote Value Growth (FY25–26)** | **11.9%** | Reserve Bank of India (RBI, 2026) *Annual Report 2025–26* |
| **Monthly UPI Volume / Value (Oct 2026)** | **25.47B / ₹31.53L Cr** | NPCI (2026) *UPI Product Statistics* (Oct 2024: 16.58B / ₹23.49L Cr) |
| **Adult Account Ownership in India (2021 $\rightarrow$ 2024)** | **77.5% $\rightarrow$ 89.0%** | World Bank Global Findex (2021 p. 25; 2025 India Microdata) |
| **Share of Inactive Accounts in India (2021)** | **35.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 32) |
| **Banked Adults Paying Merchants Only in Cash (2021)** | **670 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Banked Women Paying Merchants Only in Cash (2021)** | **350 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Adult Mobile Phone Ownership (2021 $\rightarrow$ 2024)** | **65.6% $\rightarrow$ 66.5%** | World Bank Global Findex India Microdata (Weighted) |
| **Adults Making Digital Merchant Payments (2021 $\rightarrow$ 2024)** | **11.9% $\rightarrow$ 18.3%** | World Bank Global Findex India Microdata (Weighted) |
| **Wage Earners Paid Cash-Only (2021 $\rightarrow$ 2024)** | **39.9% $\rightarrow$ 57.4%** | World Bank Global Findex India Microdata (`receive_wages == 2`) |
| **Stated Barrier (`fin25e4`): Habit / "Used to it"** | **77.4%** *(78.6% all)* | World Bank Global Findex 2025 (2024 India Microdata, Capable / All) |
| **Stated Barrier (`fin25e4`): Merchant Accepts Cash Only** | **9.8%** *(9.0% all)* | World Bank Global Findex 2025 (2024 India Microdata, Capable / All) |
| **Stated Barrier (`fin25e4`): Distrust & Scam Anxiety** | **6.5%** *(4.4% all)* | World Bank Global Findex 2025 (2024 India Microdata, Capable / All) |
| **Stated Barrier (`fin25e4`): Higher Digital Costs** | **2.3%** *(2.2% all)* | World Bank Global Findex 2025 (2024 India Microdata, Capable / All) |

---

## 📂 Project Architecture

```
.
├── Capable-but-Cash_Experiential_Learning_Phase_I.pdf   # Phase I Project Presentation
├── LITERATURE_REVIEW.md                                 # Systematic literature review (36 peer-reviewed citations)
├── NICHE_PIVOT_ANALYSIS.md                             # Critical peer review & 3 publishable niche pivots
├── README.md                                           # Project governance & execution guide
├── data/
│   ├── raw/                                            # Raw World Bank Global Findex microdata (.dta / .csv)
│   └── processed/                                      # Filtered "Capable" cohort microdata
├── figures/                                            # Model performance and interpretability outputs
├── paper/
│   ├── RESEARCH_PAPER.md                               # Complete research paper manuscript (100% cited)
│   └── paper.tex                                       # IEEE-format conference manuscript for Overleaf
└── src/
    ├── data_loader.py                                  # Strict loader supporting 2021 & 2025 Findex microdata
    ├── preprocessor.py                                 # Cohort screening (Account & Phone & Internet)
    ├── models.py                                       # Survey-weighted Logit & regularized XGBoost
    ├── explainability.py                               # TreeSHAP game-theoretic interpretability engine
    └── main.py                                         # Master pipeline orchestrator
```

---

## 🚀 Running the Empirical Pipeline on Official Data

Due to World Bank privacy terms, official survey microdata must be downloaded directly by the researcher from the World Bank Microdata Library:
1. **Option A (2025 Edition / 2024 wave):** Visit [India Global Findex 2025](https://microdata.worldbank.org/index.php/catalog/7916) (`IND_2024_FINDEX_v02_M`) and download `Findex_Microdata_2025_updateIndia.csv` / `.dta`.
2. **Option B (2021 Benchmark):** Visit [India Global Findex 2021](https://microdata.worldbank.org/index.php/catalog/4653) (`IND_2021_FINDEX_v02_M`) and download `micro_ind.csv` / `.dta`.
3. Place the file into `data/raw/`.
4. Run the master pipeline:
   ```bash
   python3 src/main.py
   ```
   *The data loader automatically detects both 2021 and 2025 formats, strictly verifies authentic files, and prevents any synthetic execution.*

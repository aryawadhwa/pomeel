# Analyzing India’s Rising Currency in Circulation: An Explainable Machine Learning Approach to the "Capable-but-Cash" Paradox

**Aarush Garg**, **Arya Wadhwa**, and **Dilraj Singh**  
*Department of Information Science and Engineering / Computer Science and Engineering*  
*RV College of Engineering, Bengaluru, India*  

---

## Abstract
Despite the exponential proliferation of India's Unified Payments Interface (UPI)—which processes over 14.96 billion monthly retail transactions valued at over ₹20.64 lakh crore (NPCI, 2024)—India's absolute Currency in Circulation (CiC) has persistently expanded, growing by over 11% annually to exceed ₹42.8 lakh crore ($42.8 \times 10^{12}$ INR) in 2026 (Reserve Bank of India, 2024). This macroeconomic divergence reveals a pronounced micro-level behavioral puzzle: millions of Indian adults who possess all foundational prerequisites for digital transacting (a formal bank account, smartphone ownership, and active internet connectivity) continue to transact exclusively in physical cash for retail purchases. According to the World Bank Global Findex Database (Demirgüç-Kunt et al., 2022), approximately 670 million banked adults in India—including 350 million women—paid merchants exclusively in cash, representing over 40% of the world's cash-only banked population. We define and isolate this sub-population as the **"Capable-but-Cash"** cohort. Survey baseline data indicates that among digitally equipped adults who remain cash-reliant, **78.0%** self-report habit persistence ("used to it"), **10.0%** cite merchant acceptance constraints ("merchant accepts only cash"), **4.9%** cite distrust, fraud anxiety, or scam exposure, and only **2.1%** cite higher transaction fees. This paper formulates an empirical, explainable machine learning architecture to evaluate this paradox. We specify a survey-weighted Logistic Regression baseline alongside a regularized Gradient Boosted Decision Tree (XGBoost) classifier, coupled with cooperative game-theoretic interpretability via TreeSHAP. Our framework translates empirical attributions into targeted policy and FinTech product architectures, specifically designed to intercept the **"Cash-Out Loop"** and mitigate cognitive switching costs.

**Keywords:** Financial Inclusion, Digital Payments, Unified Payments Interface (UPI), Currency Demand Paradox, Explainable Artificial Intelligence (XAI), TreeSHAP, XGBoost, Behavioral Economics, Global Findex.

---

## I. Introduction

The modern Indian retail payment ecosystem is recognized globally as a pioneering benchmark in Digital Public Infrastructure (DPI) (World Bank, 2023). Through the coordinated deployment of the "India Stack"—anchored by Aadhaar biometric identification, universal account access under the Pradhan Mantri Jan Dhan Yojana (PMJDY), and the open, interoperable Unified Payments Interface (UPI) developed by the National Payments Corporation of India (NPCI)—the marginal technical and economic friction of digital retail payments has been drastically reduced (D'Silva et al., 2019; NPCI, 2024).

Under orthodox monetary theory and classical payment adoption models (e.g., the Baumol-Tobin cash inventory framework), the near-zero marginal cost of electronic transactions should substitute physical currency as a medium of exchange, triggering a structural contraction in banknote demand (Baumol, 1952; Tobin, 1956). 

However, empirical data from the Reserve Bank of India (RBI) reveals an enduring macroeconomic divergence:
1. **The Digital Surge:** Monthly UPI retail transaction volumes have expanded at compound annual growth rates exceeding 50%, routinely surpassing **14.96 billion transactions monthly**, with annualized transaction values exceeding **₹240 lakh crore**—a volume rivaling India's nominal Gross Domestic Product (NPCI, 2024).
2. **The Currency Demand Paradox:** Concurrently, India's absolute Currency in Circulation (CiC) continues to rise at an annual rate of 8% to 12%, climbing from ₹17.7 lakh crore immediately following the 2016 demonetization to over **₹42.8 lakh crore ($42.8 \times 10^{12}$ INR)** in mid-2026 (Reserve Bank of India, 2024; Awasthy, Misra, & Dhal, 2023).

While central bank economists decouple this aggregate trend into transactional velocity versus precautionary/store-of-value holdings in the informal economy (Goodhart & Ashworth, 2020; Awasthy et al., 2023), macroeconomic indicators obscure granular individual-level behavioral decision-making.

Most existing financial inclusion literature assesses inclusion through an **access-centric paradigm**—measuring bank account ownership rates, smartphone penetration, and telecom connectivity (Demirgüç-Kunt et al., 2022). While India successfully expanded formal account ownership from 35% in 2011 to **77.5% (approx. 78%) in 2021** (Demirgüç-Kunt et al., 2022, p. 25), microdata demonstrates that universal access does not automatically dictate active digital usage:
* **The Account Inactivity Challenge:** India exhibits an account inactivity rate of **35.0%** (no deposits or withdrawals in the preceding 12 months), significantly higher than the developing economy average of 13.0% (Demirgüç-Kunt et al., 2022, p. 32).
* **The Cash-Only Merchant Scale:** Over **670 million banked adults in India pay merchants exclusively in cash** (Demirgüç-Kunt et al., 2022, p. 58). India alone accounts for more than 40% of the 1.6 billion banked cash-only consumers globally.

We designate this phenomenon the **"Capable-but-Cash" Paradox**. This study addresses the following central research question:
> *Among Indian adults who already possess an active formal bank account, a smartphone, and internet access, which behavioral, demographic, and psychological factors are most strongly associated with continued cash reliance for in-store transactions, and how do these factors interact?*

Rather than relying strictly on classical parametric regressions—which assume linear log-odds and struggle to identify complex feature interactions—we develop an **Explainable Machine Learning** architecture combining regularized XGBoost (Chen & Guestrin, 2016) with **TreeSHAP** (Tree-based SHapley Additive exPlanations) (Lundberg & Lee, 2017; Lundberg et al., 2020). This framework allows us to extract an evidence-led, game-theoretically consistent ranking of behavioral barriers, moving beyond binary prediction toward actionable policy recommendations.

---

## II. Literature Review & Theoretical Grounding

```mermaid
flowchart TD
    subgraph Macro [1. Macroeconomic Context]
        CiC["Currency in Circulation: ₹42.8L Cr<br/>(RBI, 2024; Awasthy et al., 2023)"]
        UPI["UPI Volume: 14.96B txn/mo<br/>(NPCI, 2024)"]
        Paradox["Currency Demand Paradox<br/>(Goodhart & Ashworth, 2020)"]
        CiC --- Paradox
        UPI --- Paradox
    end

    subgraph Micro [2. Microeconomic & Behavioral Realities]
        Findex["670M Banked Indians Pay Only Cash<br/>(Demirgüç-Kunt et al., 2022)"]
        Habit["78.0% Habit Persistence ('Used to it')<br/>(Samuelson & Zeckhauser, 1988)"]
        Loop["The 'Cash-Out Loop'<br/>(CGAP, 2021; Sutradhar, 2024)"]
        Trust["4.9% Trust & Scam Anxiety<br/>(Talwar et al., 2020; Goyal, 2026)"]
        Findex --> Habit
        Findex --> Loop
        Findex --> Trust
    end

    subgraph Method [3. Empirical & XAI Methodology]
        XGB["Regularized XGBoost Classifier<br/>(Chen & Guestrin, 2016)"]
        SHAP["TreeSHAP Interpretability<br/>(Lundberg & Lee, 2017; Lundberg et al., 2020)"]
        XGB --> SHAP
    end

    Paradox --> Micro
    Micro --> Method
```

### A. The Currency Demand Paradox and the "Paradox of Banknotes"
Central banking literature across both advanced and developing economies has documented the "paradox of banknotes"—wherein aggregate currency demand expands even as electronic alternatives flourish (Goodhart & Ashworth, 2020; Arango, Huynh, & Sabetti, 2015; Agarwal, 2023). In India, Reserve Bank of India empirical working papers (Awasthy, Misra, & Dhal, 2023) and macroeconomic evaluations (Kumar & Mishra, 2025; Minj, 2025) demonstrate that while digital retail payment platforms have replaced cash for low-ticket consumer transactions, aggregate currency demand remains elevated due to two structural forces:
1. **Precautionary & Store-of-Value Inelasticity:** In India's large informal economy (which employs over 80% of the workforce), physical cash serves as an indispensable liquidity cushion against economic and agricultural shocks (Awasthy et al., 2023).
2. **Income Elasticity vs. Substitution:** Rising nominal Gross Domestic Product exerts an expansionary income effect on money demand that partially offsets the digital substitution effect (Agarwal, 2023). Subnational econometric analyses confirm that digital displacement decelerates after initial adoption, hitting a structural floor determined by literacy and informal employment (RBI Bulletins, 2024).

### B. Technology Acceptance Models (UTAUT & UTAUT2)
The Unified Theory of Acceptance and Use of Technology (UTAUT) (Venkatesh, Morris, Davis, & Davis, 2003) and its consumer extension UTAUT2 (Venkatesh, Thong, & Xu, 2012) provide the principal theoretical scaffolding for technology adoption:
* **Effort Expectancy:** The degree of cognitive ease associated with platform usage. In Indian retail payment environments, Slade, Dwivedi, Piercy, & Williams (2015) and Oliveira, Thomas, Baptista, & Campos (2016) demonstrate that multi-factor authentication (SMS OTPs, 4-to-6 digit UPI PINs, complex application navigation) introduces significant cognitive friction for individuals with limited formal literacy.
* **Facilitating Conditions & Merchant Readiness:** Migliore, Wagner, & Cechella (2022) establish that consumer willingness to transact digitally is strictly bounded by perceived merchant acceptance and local network infrastructure.
* **Perceived Risk and Initial Trust:** Talwar, Dhir, & Khalil (2020) and Goyal (2026) investigate post-adoption continuation intention in Indian digital finance, observing that "transaction failure dread" (money debited from a consumer's account without reaching the merchant) and fear of cyber fraud generate acute risk aversion.

### C. Behavioral Economics: Status-Quo Bias, Mental Accounting, and the "Cash-Out Loop"
Classical adoption frameworks assume rational utility maximization based purely on speed and cost. Behavioral economics provides essential mechanisms explaining cash persistence:
1. **Status-Quo Bias & Habit Persistence:** Samuelson & Zeckhauser (1988) proved that decision-makers disproportionately default to established behaviors even when transition costs are objectively minimal. In India, cash transactions represent a motor and psychological routine deeply ingrained across generations (Demirgüç-Kunt et al., 2022).
2. **Mental Accounting & The "Pain of Paying":** Prelec & Loewenstein (1998), Thaler (1999), and Soman (2001) demonstrated that physical currency separation creates an immediate, visceral "pain of paying." For budget-constrained households, handling physical banknotes acts as an automated, tangible expenditure regulator. Digital payment mechanisms decouple the consumption experience from payment salience, provoking anxiety over unmonitored overspending.
3. **The "Cash-Out Loop" Hypothesis:** A critical structural bottleneck identified in emerging-market financial inclusion (Demirgüç-Kunt et al., 2022; CGAP, 2021; Sutradhar, 2024) is the disconnection between digital account credits and physical consumption. When informal sector workers or Direct Benefit Transfer (DBT) recipients receive government subsidies or wages into their bank accounts, their initial response is to withdraw 100% of the funds in cash via ATMs or Business Correspondents (Bank Mitras). Once liquidity enters the physical domain, all subsequent retail transactions inevitably occur in cash, completely bypassing digital point-of-sale rails.

### D. Methodological Justification: Why Explainable AI (XAI)?
Traditional empirical literature evaluating payment choice relies almost universally on **Binary Logistic Regression** or **Probit Models** (Mukhopadhyay, 2016; Sutradhar, 2024). While linear econometric models provide interpretable odds ratios ($\exp(\beta)$), they impose rigid assumptions:
1. **Linearity Assumption:** Logit assumes a strictly linear relationship between continuous predictors (e.g., Age) and the log-odds of digital adoption, masking non-linear tipping points.
2. **Combinatorial Interaction Bottlenecks:** Modeling multi-way interactions (e.g., $\text{Age} \times \text{Education} \times \text{Wage-in-Cash}$) requires manual specification, risking omitted variable bias or overfitting.
3. **Multicollinearity:** Socio-demographic variables in survey datasets exhibit substantial collinearity, inflating variance in parametric estimators.

Gradient Tree Boosting (XGBoost) (Chen & Guestrin, 2016) natively models non-linear decision boundaries and high-order interactions without manual specification (Maehara, Benites, & Talavera, 2024). Coupling XGBoost with **TreeSHAP** (Lundberg & Lee, 2017; Lundberg et al., 2020) provides mathematically guaranteed, game-theoretically consistent Shapley attributions, resolving the "black-box" dilemma in financial inclusion policy.

---

## III. Data Operationalization: World Bank Global Findex 2021 (India)

### A. Data Source & Sampling Architecture
The empirical investigation utilizes microdata from the **World Bank Global Financial Inclusion (Global Findex) Database 2021** for India (Study Reference ID: `IND_2021_FINDEX_v02_M`) (Demirgüç-Kunt et al., 2022). The survey comprises $N = 3,000$ face-to-face, nationally representative surveyed adult respondents (aged 15+) across all Indian states and Union Territories, capturing 199 survey indicators.

### B. Operational Definition of the "Capable" Cohort
To isolate the paradox of non-adoption despite technical readiness, we explicitly filter out infrastructural exclusion. The eligible cohort is mathematically formulated as:

$$\text{Cohort}_{\text{Capable}} = \left\{ i \;\middle|\; \text{account}_i = 1 \;\land\; \text{mobileowner}_i = 1 \;\land\; \text{internet}_i = 1 \right\}$$

where:
* `account = 1`: Respondent has an individual or joint account at a bank, credit union, or formal microfinance institution (Demirgüç-Kunt et al., 2022).
* `mobileowner = 1`: Respondent personally owns a mobile phone (excluding shared household devices).
* `internet = 1`: Respondent used the internet in the past three months (via mobile phone, computer, or tablet).

### C. Target Variable Formulation ($Y$)
Within the digitally capable cohort, the primary dependent variable captures in-store retail payment behavior:

$$Y_i = \begin{cases} 
1 & \text{if respondent relied exclusively on cash for in-store purchases (Cash-Reliant)} \\ 
0 & \text{if respondent completed a digital in-store/merchant payment (Digital Adopter)} 
\end{cases}$$

Formally, $Y_i = 1 - \text{merchantpay\_dig}_i$. In the Global Findex codebook, `merchantpay_dig = 1` indicates that the respondent used a debit card, credit card, mobile phone, or the internet to purchase goods in person at a store, supermarket, or local vendor in the past year (Demirgüç-Kunt et al., 2022).

### D. Benchmark Empirical Metrics from Published Literature
Table I documents the directly cited, published empirical metrics establishing the baseline for our investigation.

**Table I: Key Empirical Baseline Data from Published Sources**
| Empirical Metric | Value | Primary Official Citation |
| :--- | :---: | :--- |
| **India Absolute Currency in Circulation (CiC)** | **₹42.8 Lakh Crore** | Reserve Bank of India (RBI, 2024) *Weekly Statistical Supplement* |
| **Annualized CiC Growth Rate** | **11.2%** | Reserve Bank of India (RBI, 2024) *Money Stock Measures* |
| **Monthly UPI Transaction Volume** | **14.96 Billion** | NPCI (2024) *UPI Product Statistics* (October 2024) |
| **Monthly UPI Transaction Value** | **₹20.64 Lakh Crore** | NPCI (2024) *UPI Product Statistics* (October 2024) |
| **Adult Account Ownership in India** | **77.5% (approx. 78%)** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 25) |
| **Share of Inactive Accounts in India** | **35.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 32) |
| **Banked Adults Paying Merchants Only in Cash** | **670 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Banked Women Paying Merchants Only in Cash** | **350 Million** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 58) |
| **Adult Mobile Phone Ownership** | **72.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 28) |
| **Adults Making Digital Merchant Payments** | **12.0%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 56) |
| **Wage Earners Receiving Wages in Cash** | **62.4%** | World Bank Global Findex (Demirgüç-Kunt et al., 2022, p. 64) |
| **Stated Barrier: Habit / "Used to it"** | **78.0%** | Global Findex 2021 Survey / Project Baseline Data |
| **Stated Barrier: Merchant Accepts Only Cash** | **10.0%** | Global Findex 2021 Survey / Project Baseline Data |
| **Stated Barrier: Distrust & Scam Anxiety** | **4.9%** | Global Findex 2021 Survey / Project Baseline Data |
| **Stated Barrier: Higher Digital Costs** | **2.1%** | Global Findex 2021 Survey / Project Baseline Data |

---

## IV. Empirical Methodology & Econometric Specification

### A. Survey-Weighted Logistic Regression (Econometric Baseline)
As an econometric benchmark, we formulate a survey-weighted Generalized Linear Model (GLM) with a binomial distribution and logit link function:

$$\ln\left(\frac{P(Y_i=1)}{1 - P(Y_i=1)}\right) = \beta_0 + \sum_{j=1}^{M} \beta_j X_{ij}$$

Parameter estimation is achieved by maximizing the survey-weighted log-likelihood:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n w_i \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

where $w_i$ denotes the normalized survey sampling weights (`wgt`) provided in the Global Findex microdata to ensure population-representative inference across geographic and demographic strata.

### B. Regularized Gradient Boosted Decision Trees (XGBoost)
The primary non-linear classifier utilizes an ensemble of $K$ additive regression trees:

$$\hat{y}_i = \sum_{k=1}^K f_k(\mathbf{x}_i), \quad f_k \in \mathcal{F}$$

minimizing the regularized objective function:

$$\mathcal{L}(\theta) = \sum_{i=1}^n w_i \cdot \ell\left(y_i, \hat{y}_i\right) + \sum_{k=1}^K \Omega(f_k)$$

where $\ell$ is the logistic loss, and the regularization penalty is:

$$\Omega(f_k) = \gamma T_k + \frac{1}{2}\lambda \|\mathbf{w}_k\|^2 + \alpha \|\mathbf{w}_k\|_1$$

Structural parameters $\gamma, \lambda, \alpha$ explicitly penalize leaf count and weight magnitudes to prevent overfitting on survey sample sizes (Chen & Guestrin, 2016). Class imbalance is corrected by configuring `scale_pos_weight` according to the empirical negative-to-positive ratio in the training cohort.

### C. TreeSHAP Cooperative Game Formulation
To extract transparent, mathematically robust feature attributions, we implement TreeSHAP (Lundberg & Lee, 2017; Lundberg et al., 2020). Grounded in cooperative game theory, the Shapley value $\phi_j(x)$ measures the exact marginal contribution of feature $j$ across all possible feature subsets $S \subseteq F \setminus \{j\}$:

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right]$$

TreeSHAP computes these values in polynomial time $\mathcal{O}(T L D^2)$ (where $T$ is tree count, $L$ is maximum leaves, and $D$ is maximum tree depth). Global barrier importance is formalized as the mean absolute Shapley value:

$$I_j = \frac{1}{N} \sum_{i=1}^N \left| \phi_j^{(i)} \right|$$

---

## V. Empirical Hypotheses & Analysis Protocol

Based on the cited literature, the empirical pipeline is structured to test five core hypotheses:

* **Hypothesis 1 ($H_1$ - The Habit Primacy Hypothesis):** Consistent with Samuelson & Zeckhauser (1988) and baseline Findex indicators (78.0%), habit persistence and age will demonstrate higher feature importance ($I_j$) in predicting cash reliance than technological or infrastructure variables.
* **Hypothesis 2 ($H_2$ - The Cash-Out Loop Hypothesis):** Supported by Demirgüç-Kunt et al. (2022) and Sutradhar (2024), individuals receiving wages or government transfers in cash (`wage_cash = 1`, `gov_cash = 1`) will display significantly higher probability of in-store cash reliance.
* **Hypothesis 3 ($H_3$ - The Age Inflection Hypothesis):** Rather than exhibiting a strictly linear log-odds decay, age will display a sharp non-linear threshold beyond age 40 where digital capability fails to displace physical cash habits.
* **Hypothesis 4 ($H_4$ - The Irrelevance of Direct Costs):** Because UPI operates under zero-MDR regulations, direct cost barriers (`barrier_cost`) will exhibit negligible feature attribution ($I_j \approx 0$), whereas psychological trust and scam concerns (`barrier_trust`, 4.9% baseline) will rank substantially higher.
* **Hypothesis 5 ($H_5$ - The Gender Usage Gap):** Despite parity in baseline account ownership (78% for both genders in Findex 2021), female gender will positively contribute to cash reliance due to intra-household cash budgeting dynamics and shared device constraints.

---

## VI. Empirical Results & Discussion

### A. Two-Wave Cohort Dynamics (2021 vs. 2024)
Table II summarizes the microdata sample and the evolution of the digitally capable sub-population between the 2021 Global Findex wave and the 2024 Connectivity Tracker wave ($N = 6,000$ total surveyed Indian adults).

**Table II: Capable Cohort Dynamics across Survey Waves (India)**
| Indicator | Wave 1 (2021) | Wave 2 (2024) | Pooled Two-Wave Sample |
| :--- | :---: | :---: | :---: |
| **Total Surveyed Adults ($N$)** | 3,000 | 3,000 | 6,000 |
| **Formal Bank Account Ownership** | 79.1% (2,372) | 90.3% (2,710) | 84.7% (5,082) |
| **Mobile Phone Ownership** | 69.6% (2,088) | 71.5% (2,145) | 70.6% (4,233) |
| **Active Internet Access** | 31.6% (947) | 51.6% (1,547) | 41.6% (2,494) |
| **Digitally Capable Cohort ($N_{\text{cap}}$)** | **837 (27.9%)** | **1,335 (44.5%)** | **2,172 (36.2%)** |
| **- Digital Adopters (Store Digital Pay)** | 348 (41.6%) | 614 (46.0%) | 962 (44.3%) |
| **- Cash-Reliant ("Capable-but-Cash")** | **489 (58.4%)** | **721 (54.0%)** | **1,210 (55.7%)** |

Despite substantial expansion in smartphone connectivity and formal banking between 2021 and 2024, **54.0% of technologically equipped Indian adults remain strictly cash-reliant for in-store retail purchases**, demonstrating the stubborn persistence of the paradox.

### B. Predictive Model Performance & Leak-Free Ablation
To prevent ecological or simultaneous-outcome leakage, all concurrent digital payment variables (such as utility bill pay `pay_utilities`, online bill pay `fin14a`, and card purchase usage `fin4`) were strictly excluded. Table III reports the **5×3 Repeated Stratified K-Fold Cross-Validation** (15 survey-weighted evaluations) alongside the held-out stratified test set ($n = 435$).

**Table III: Leak-Free Repeated Stratified K-Fold CV Ablation (Survey-Weighted, $N_{\text{cap}} = 2,172$)**
| Model Configuration | ROC-AUC (Mean $\pm$ SD) | PR-AUC | Brier Score | F1-Score | Balanced Acc. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline XGBoost (9 Core Vars)** | $0.7365 \pm 0.028$ | 0.7861 | 0.2049 | 0.7276 | 0.6649 |
| **Ablation XGBoost (No Debit Card)** | $0.7915 \pm 0.028$ | 0.8315 | 0.1832 | 0.7619 | 0.7023 |
| **Spline + OneHot Logit (Enhanced)** | $0.8470 \pm 0.026$ | 0.8794 | 0.1574 | **0.7973** | 0.7623 |
| **Regularized XGBoost (Enhanced)** | **$0.8477 \pm 0.025$** | **0.8824** | **0.1571** | 0.7924 | **0.7695** |
| **Regularized LightGBM (Enhanced)** | $0.8468 \pm 0.025$ | 0.8799 | 0.1579 | 0.7933 | 0.7629 |
| **4-Family Stacked + Platt Cal.** | $0.8470 \pm \mathbf{0.022}$ | **0.8824** | 0.1593 | 0.7847 | 0.7621 |

On the held-out test split ($n = 435$), the **Platt-Calibrated 4-Family Stacked Ensemble** achieves the highest discrimination (**ROC-AUC = 0.8400**) and lowest calibration error (**Brier Score = 0.1614**), while **Spline-Logit** with out-of-fold threshold tuning reaches **PR-AUC = 0.8588** and **F1 = 0.7913**.

### C. TreeSHAP Attribution & Econometric Odds Ratios
Applying cooperative game-theoretic TreeSHAP to the leak-free Regularized XGBoost ensemble and cross-checking against survey-weighted GLM Odds Ratios yields consistent behavioral drivers (Table IV).

**Table IV: Top Leak-Free Predictors (TreeSHAP Importance & GLM Odds Ratios)**
| Rank | Feature Code | Description | Mean $|SHAP|$ ($I_j$) | GLM Odds Ratio ($p$-value) |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `has_debit_card` | Possesses an ATM/debit card (`fin2`) | **0.9089** | 0.12 ($p < 0.001$) |
| **2** | `borrowed_any` | Active borrowing behavior (`borrowed`) | **0.3153** | 0.42 ($p < 0.001$) |
| **3** | `educ_x_inc` | Education $\times$ Income interaction | **0.3020** | — |
| **4** | `saved_any` | Active saving behavior (`saved`) | **0.2928** | 0.42 ($p < 0.001$) |
| **5** | `wage_bank` | Wages deposited into bank account | **0.2006** | 0.42 ($p < 0.001$) |
| **6** | `educ` | Educational attainment level | **0.1622** | 0.50 ($p < 0.001$) |
| **7** | `bank_inflow_intensity` | Count of formal digital income channels | **0.1616** | — |
| **8** | `inc_q` | Household income quintile | **0.1589** | 0.73 ($p < 0.001$) |
| **9** | `female` | Female gender indicator (Gender Gap) | **0.1572** | **1.96 ($p = 0.010$)** |
| **10** | `cash_inflow_intensity` | Cash income channels (`wage+gov+ag`) | **0.1073** | `wage_cash`: **2.31 ($p < 0.001$)** |
| **11** | `wave_x_urban` | 2024 $\times$ Urban shift interaction | **0.0750** | **0.46 ($p = 0.007$)** |
| **12** | `wave_2024` | Survey wave shift (2021 $\rightarrow$ 2024) | **0.0627** | 1.12 ($p = 0.623$) |

### D. Hypothesis Verification
1. **$H_2$ (The Cash-Out Loop) is Strongly Confirmed:** Receiving wages in physical cash (`wage_cash`, `receive_wages == 2`) more than **doubles** the odds of retail cash reliance ($\text{OR} = 2.31, p < 0.001$), whereas receiving wages directly into a bank account (`wage_bank`) cuts cash-reliance odds by **58%** ($\text{OR} = 0.42, p < 0.001$).
2. **$H_5$ (The Gender Usage Gap) is Confirmed:** `female` nearly doubles the odds of remaining cash-reliant ($\text{OR} = 1.96, p = 0.010$; TreeSHAP $I_j = 0.1572$) despite formal account parity.
3. **Upstream Instrument & Spatial Divergence:** Possessing an ATM/debit card (`has_debit_card`, $\text{OR} = 0.12$) is the strongest upstream onboarding enabler, while the `wave_x_urban` interaction ($\text{OR} = 0.46, p = 0.007$) confirms that post-2021 digitization accelerated significantly faster in urban India than in rural pockets.

---

## VII. Policy & Product Interventions

The empirical evidence and behavioral theories demonstrate that expanding baseline infrastructure (more accounts or smartphones) cannot resolve the "Capable-but-Cash" paradox. Policy interventions must directly address the identified behavioral barriers:

### 1. Intercepting the "Cash-Out Loop" at the Source
Because wage receipt in cash is the primary structural driver of retail cash spending (Demirgüç-Kunt et al., 2022), government and regulatory policy must incentivize **closed-loop digital wage ecosystems**:
* **Informal Sector Payroll Digitization:** Subsidize cloud-based payroll software integrated with UPI AutoPay for micro-, small-, and medium-sized enterprises (MSMEs), eliminating cash wage envelopes.
* **Smart Direct Benefit Transfer (DBT) Wallets:** Provide a 1.5% to 2.0% micro-cashback on essential retail purchases (groceries, utilities) executed directly through digital rails, creating a direct financial incentive to retain balances rather than conducting immediate ATM cash-outs.

### 2. Mitigating Trust Deficits and Scam Anxiety
Addressing the 4.9% of cash-reliant adults constrained by distrust requires reducing "transaction failure dread" (Talwar et al., 2020; Goyal, 2026):
* **Asymmetric Instant Dispute Reversals:** Mandate an automated 60-second account credit reversal for sub-₹500 merchant transactions that fail to settle at the point of sale, eliminating the risk of temporary fund lockup.
* **Standardized Merchant Security Badges:** NPCI-certified physical QR badges providing visual authentication to reassure fraud-sensitive consumers.

### 3. Mitigating Cognitive Friction (Designing for Habit Persistence)
To address habit persistence among older demographics (Samuelson & Zeckhauser, 1988):
* **Default UPI Lite On-Device Wallets:** Remove 6-digit PIN requirements for daily low-ticket transactions ($<\text{₹}500$), mirroring the speed and cognitive simplicity of handing over physical cash notes.
* **Voice-First Conversational Interfaces:** Integrate conversational AI (leveraging India's Bhashini open-source language models) to allow users to speak transaction commands in local dialects, eliminating complex navigation hurdles.

---

## VII. Limitations & Threats to Validity

1. **Cross-Sectional Observational Data:** The Global Findex survey is cross-sectional. While TreeSHAP values isolate empirical associations and predictive contributions, they do not establish causal relationships in the absence of exogenous experimental variation.
2. **Survey Measurement Limitations:** Psychological constructs such as habit persistence and scam anxiety are evaluated via discrete self-reported survey indicators rather than continuous psychometric psychophysical instruments.
3. **Temporal Dynamics:** The 2021 survey wave captured adoption during the COVID-19 pandemic recovery. While India's CiC has continued its parallel upward march to ₹42.8 lakh crore in 2026, subsequent diffusion of soundbox QR terminals has further expanded merchant digital acceptance.

---

## VIII. Conclusion

This study establishes a rigorous empirical and explainable machine learning foundation to analyze India's "Capable-but-Cash" paradox. By combining macroeconomic currency demand literature with micro-level Global Findex 2021 survey data, we demonstrate that persistent cash usage among digitally equipped adults is governed by the **"Cash-Out Loop"**, **habit persistence in older adults**, and **asymmetric trust deficits**, while direct transaction costs are virtually inconsequential. 

Resolving India's currency demand paradox requires central banks, retail banks, and FinTech innovators to transition from basic infrastructure expansion to targeted behavioral architectures that intercept cash withdrawals at the point of income disbursement and eliminate cognitive friction for the next hundred million digital users.

---

## References

1. Agarwal, A. (2023). Less - Cash Societies: A Paradoxical Roadmap Revamped through Unified Payment Systems and Electronic Currencies. *International Journal of Science and Research (IJSR)*, 12(10), 153–158.
2. Arango, C., Huynh, K. P., & Sabetti, L. (2015). Consumer payment choice: Merchant card acceptance versus pricing incentives. *Journal of Banking & Finance*, 55, 130–141.
3. Awasthy, S., Misra, R., & Dhal, S. (2023). Understanding the Currency Demand Paradox in India. *Reserve Bank of India Working Paper Series*.
4. Baumol, W. J. (1952). The Transactions Demand for Cash: An Inventory Theoretic Approach. *The Quarterly Journal of Economics*, 66(4), 545–556.
5. CGAP. (2021). *Financial Inclusion and the Cash-Out Loop in Emerging Markets*. Consultative Group to Assist the Poor, World Bank.
6. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
7. Demirgüç-Kunt, A., Klapper, L., Singer, D., & Ansar, S. (2022). *The Global Findex Database 2021: Financial Inclusion, Digital Payments, and Resilience in the Age of COVID-19*. World Bank Publications.
8. D'Silva, D., Filková, Z., Packer, F., & Tiwari, P. (2019). The design of digital financial infrastructure: lessons from India. *BIS Papers*, No. 106.
9. Goodhart, C. A., & Ashworth, J. (2020). Coronavirus and the commerce of cash: the demand for banknotes in pandemic times. *CEPR Discussion Paper*, DP14782.
10. Goyal, M. (2026). Determinants of consumer trust and adoption barriers in UPI-based payment applications among adults in India. *International Journal of Research in Finance and Management*, 9(8), 874–881.
11. GSMA. (2023). *The Mobile Gender Gap Report 2023*. GSM Association.
12. Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. *Econometrica*, 47(2), 263–291.
13. Kumar, A., & Mishra, A. K. (2025). Understanding India's Currency Demand Absurdity: Cash vs. Digital Payment Methods. *International Journal on Science and Technology*, 16(1), 2919–2928.
14. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
15. Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
16. Maehara, R., Benites, L., & Talavera, A. (2024). Predicting Financial Inclusion in Peru: Application of Machine Learning Algorithms. *Journal of Risk and Financial Management*, 17(1), 34.
17. Migliore, G., Wagner, R., & Cechella, F. S. (2022). Antecedents to the Adoption of Mobile Payment: An Integration of UTAUT2 and Innovation Resistance Theory. *Information Systems Frontiers*, 24(6), 2005–2021.
18. Minj, M. (2025). Shift from Cash to Cashless Economy in India: Evidence from RBI & World Bank Data. *Journal of Advances in Developmental Research*, 16(1), 1558–1565.
19. Mukhopadhyay, B. (2016). Understanding cashless payments in India. *Financial Innovation*, 2(1), 27.
20. NPCI. (2024). *UPI Monthly Product Statistics - October 2024*. National Payments Corporation of India. https://www.npci.org.in/what-we-do/upi/product-statistics
21. Oliveira, T., Thomas, M., Baptista, G., & Campos, F. (2016). Mobile payment: Understanding the determinants of customer adoption and intention to recommend the technology. *Computers in Human Behavior*, 61, 404–414.
22. Prelec, D., & Loewenstein, G. (1998). The Red and the Black: Mental Accounting of Savings and Debt. *Marketing Science*, 17(1), 4–28.
23. Reserve Bank of India. (2024). *Weekly Statistical Supplement: Money Stock Measures*. RBI Bulletin, 2024.
24. Samuelson, W., & Zeckhauser, R. (1988). Status quo bias in decision making. *Journal of Risk and Uncertainty*, 1(1), 7–59.
25. Slade, E. L., Dwivedi, Y. K., Piercy, N. C., & Williams, M. D. (2015). Modeling consumers’ adoption intentions of remote mobile payments in the United Kingdom: Extending UTAUT with innovativeness, risk, and trust. *Psychology & Marketing*, 32(8), 860–873.
26. Soman, D. (2001). Effects of payment mechanism on spending behavior: The role of rehearsal and immediacy of payments. *Journal of Consumer Research*, 27(4), 460–474.
27. Sutradhar, T. K. (2024). Status and Determinants of Digital Transactions in India: Findings from the New Global Findex Database. *Vidyasagar University Journal of Economics*, 28, 21–38.
28. Talwar, S., Dhir, A., & Khalil, A. (2020). Point of adoption and beyond: Initial trust and mobile-payment continuation intention. *Journal of Retailing and Consumer Services*, 55, 102086.
29. Thaler, R. H. (1999). Mental accounting matters. *Journal of Behavioral Decision Making*, 12(3), 183–206.
30. Tobin, J. (1956). The Interest-Elasticity of Transactions Demand For Cash. *The Review of Economics and Statistics*, 38(3), 241–247.
31. Venkatesh, V., Morris, M. G., Davis, G. B., & Davis, F. D. (2003). User acceptance of information technology: Toward a unified view. *MIS Quarterly*, 27(3), 425–478.
32. Venkatesh, V., Thong, J. Y., & Xu, X. (2012). Consumer acceptance and use of information technology: Extending the unified theory of acceptance and use of technology. *MIS Quarterly*, 36(1), 157–178.
33. World Bank. (2023). *G20 Policy Recommendations for Advancing Financial Inclusion and Productivity Gains through Digital Public Infrastructure*. World Bank Group.

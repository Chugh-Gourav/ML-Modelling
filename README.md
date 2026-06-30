# 📊 Bayesian Media Mix Model (MMM)

A production-quality **Media Mix Model** built with **PyMC** (Bayesian MCMC) to measure the true incremental value of marketing channels and optimise budget allocation.

This project was built as an end-to-end learning exercise and engineering knowledge-sharing resource — from raw data wrangling through to calibrated ROAS insights and a product vision for operationalising MMM at scale.

---

## 🎯 Why This Exists

| Problem | Solution |
|---|---|
| Raw data was monthly (only 12 rows) — far too small for regression | Interpolated to **weekly** granularity (~52 data points) |
| Standard OLS regression produced **negative coefficients** (e.g. Affiliates, Content Marketing appearing to *hurt* sales) | Bayesian model with **HalfNormal priors** enforces non-negative media weights |
| No modelling of carry-over or saturation effects | Engineered **Adstock** (geometric decay) and **Diminishing Returns** (log transform) features |
| Raw model ROAS is correlation, not causation | Documented a **Calibration Framework** using Geo-Holdout incrementality experiments |

---

## 📂 Repository Structure

```
.
├── Data/
│   ├── Raw Data/               # Source CSVs (large files gitignored — see below)
│   ├── Weekly_MMM_Data.csv     # Cleaned weekly dataset (output of Step 1)
│   └── Engineered_MMM_Data.csv # Adstock + Log transformed features (output of Step 2)
│
├── Model/
│   ├── prepare_weekly_data.py       # Step 1: Aggregate daily sales + interpolate monthly media → weekly
│   ├── feature_engineering_mmm.py   # Step 2: Apply Adstock decay & log saturation transforms
│   ├── train_baseline_mmm.py       # Step 3: OLS baseline (demonstrates multicollinearity problem)
│   ├── train_bayesian_mmm.py       # Step 4: Bayesian PyMC model (the real model)
│   ├── evaluate_mmm.py             # Step 5: ROAS calculation & channel ranking
│   ├── export_excel.py             # Utility: export weekly data to Excel
│   └── export_engineered_excel.py  # Utility: export engineered data to Excel
│
├── Insights/
│   ├── OLS_Regression_Results.txt   # Full statsmodels OLS summary
│   ├── Bayesian_MMM_Results.txt     # PyMC channel weights + HDI intervals
│   ├── MMM_ROAS_Insights.csv        # Final ROAS rankings per channel
│   └── Walkthrough_and_Strategy.md  # Detailed strategy document (AI PM perspective)
│
├── docs/
│   └── ENGINEERING_WALKTHROUGH.md   # 🎤 Knowledge-sharing session guide
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/media-mix-model.git
cd media-mix-model

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the Data

> **Note:** The two large raw CSV files (`firstfile.csv` ~156 MB, `Sales.csv` ~117 MB) are **gitignored** to keep the repo lightweight. The pre-processed weekly CSVs *are* included so you can jump straight to modelling.

If you have the raw files, place them in `Data/Raw Data/` and re-run Step 1.

### 3. Run the Pipeline

Each script is standalone and numbered. Run them in order:

```bash
# Step 1 — Aggregate to weekly (requires raw data files)
python Model/prepare_weekly_data.py

# Step 2 — Feature engineering (Adstock + Diminishing Returns)
python Model/feature_engineering_mmm.py

# Step 3 — OLS Baseline (demonstrates the multicollinearity problem)
python Model/train_baseline_mmm.py

# Step 4 — Bayesian MMM (the real model — takes ~1-2 min)
python Model/train_bayesian_mmm.py

# Step 5 — ROAS evaluation & channel ranking
python Model/evaluate_mmm.py
```

---

## 📈 Key Results

### Bayesian Model (R² = 0.47)

| Channel | Weight | Share of Media Contribution | ROAS |
|---|---|---|---|
| **Sponsorship** | 0.300 | 59.4% | ~$2.1M |
| **Online Marketing** | 0.146 | 16.9% | ~$1.1M |
| **SEM** | 0.117 | 8.1% | ~$1.1M |
| **Radio** | 0.101 | 1.0% | ~$2.8M *(highest efficiency)* |
| **TV** | 0.088 | 7.3% | ~$2.1M |
| **Digital** | 0.093 | 3.3% | ~$1.4M |

### Why the OLS Baseline Failed

The standard OLS model (R² = 0.57) produced **negative coefficients** for Affiliates (-$55.8M) and Content Marketing (-$24.5M), falsely implying that spending on these channels *reduced* sales. This is a textbook symptom of **multicollinearity** — the Bayesian model's HalfNormal priors solve this by constraining all media weights ≥ 0.

---

## 🧠 Core Concepts (for the Knowledge-Sharing Session)

See the full engineering walkthrough at [`docs/ENGINEERING_WALKTHROUGH.md`](docs/ENGINEERING_WALKTHROUGH.md), covering:

1. **Adstock** — Geometric decay to model the "memory" of advertising
2. **Diminishing Returns** — Log transforms to model audience saturation
3. **Bayesian Priors** — How HalfNormal priors enforce business constraints
4. **ROAS Calculation** — Top-down attribution from model weights
5. **Calibration** — Grounding modelled ROAS with incrementality experiments
6. **The Dual-Speed Loop** — Strategic (MMM) + Tactical (DDA) measurement
7. **Hands-on Exercises** — Code challenges for the audience

---

## ⚠️ Known Limitations

- **Sample size**: 52 weeks of interpolated data; production models need 2–3 years
- **No external regressors**: Missing macro-economic controls (GDP, seasonality harmonics, competitor spend)
- **Assumed Adstock alphas**: The feature engineering step uses assumed decay rates; the Bayesian model learns its own weights but not the decay rates themselves
- **Uncalibrated ROAS**: The absolute dollar figures are modelled upper-bounds — see the Calibration section in the walkthrough for how to ground-truth them

---

## 📚 References

- [Google's Modern Measurement Playbook](https://www.thinkwithgoogle.com/)
- [PyMC Documentation](https://www.pymc.io/projects/docs/en/stable/)
- [Meta's Robyn (Open-Source MMM)](https://github.com/facebookexperimental/Robyn)
- [Google's Meridian (Open-Source MMM)](https://github.com/google/meridian)

---

## 📄 License

This project is shared for **educational and knowledge-sharing purposes**. The synthetic/sample dataset is included for reproducibility. No proprietary data is exposed.

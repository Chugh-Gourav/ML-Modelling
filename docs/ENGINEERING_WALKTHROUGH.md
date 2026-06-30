# 🎤 Engineering Knowledge-Sharing Session: Bayesian Media Mix Modelling

> **Session Duration:** ~45–60 minutes  
> **Audience:** Engineers, Data Scientists, Product Managers  
> **Prerequisites:** Basic Python, familiarity with linear regression concepts  
> **Repo:** Clone this repo and install `requirements.txt` before the session

---

## 📋 Session Agenda

| # | Topic | Time | Script |
|---|---|---|---|
| 1 | The Business Problem | 5 min | — |
| 2 | Data Wrangling: Monthly → Weekly | 10 min | `prepare_weekly_data.py` |
| 3 | Feature Engineering: Adstock & Saturation | 10 min | `feature_engineering_mmm.py` |
| 4 | The OLS Baseline (and why it fails) | 10 min | `train_baseline_mmm.py` |
| 5 | The Bayesian Solution (PyMC) | 15 min | `train_bayesian_mmm.py` |
| 6 | ROAS Evaluation & Calibration | 10 min | `evaluate_mmm.py` |
| 7 | Q&A + Hands-on Exercises | 10 min | — |

---

## 1. The Business Problem (5 min)

### What is an MMM?

A **Media Mix Model** answers the question every CMO asks:

> *"I'm spending £10M across TV, Search, Social, and Sponsorships. Which channels are actually driving revenue — and where should I move the budget?"*

Unlike click-based attribution (Google Analytics, last-click), MMM works at the **aggregate statistical level** — it doesn't need cookies or user-level tracking. This makes it:
- ✅ Privacy-safe (no PII required)
- ✅ Cross-channel (can measure offline channels like TV, Radio, Out-of-Home)
- ✅ Strategic (answers "where to allocate budget" rather than "which ad to show")

### The Dataset

We have 1 year of data for an e-commerce company:
- **Daily sales** (GMV — Gross Merchandise Value) → `firstfile.csv`
- **Monthly media spend** across 9 channels → `MediaInvestment.csv`
- **Monthly NPS** (Net Promoter Score) → `MonthlyNPSscore.csv`
- **Special sale events** (holidays, flash sales) → `SpecialSale.csv`

**🔑 Key Insight for the audience:** The raw monthly data gives us only **12 rows** — you can't run a meaningful regression on 12 rows with 10+ features. This is the first real-world challenge the pipeline solves.

---

## 2. Data Wrangling: Monthly → Weekly (10 min)

📄 **Script:** [`Model/prepare_weekly_data.py`](../Model/prepare_weekly_data.py)

### What This Script Does

```
Daily Sales ─────┐
                  ├──→ Weekly Dataset (53 rows)
Monthly Media ───┤     ┌─────────────────────────┐
Monthly NPS ─────┤     │ Date | GMV | TV | ... | │
Special Sales ───┘     │ 2015-07-05 | 23M | 1.2 │
                       └─────────────────────────┘
```

### Key Engineering Decisions

1. **Sales → Weekly:** Simple `groupby` with `pd.Grouper(freq='W-SUN')` sums daily GMV per week.

2. **Media Spend → Weekly (interpolation):** Monthly spend is divided evenly across days in that month, then re-aggregated to weekly sums. This preserves the total monthly budget while creating the weekly granularity we need.

   ```python
   # Divide monthly spend by days in month → daily spend → sum to weekly
   daily_media[col] = daily_media[col] / daily_media["Days_in_Month"]
   weekly_media = daily_media.groupby(pd.Grouper(key="Date", freq="W-SUN")).sum()
   ```

3. **NPS → Weekly:** Forward-filled from monthly to daily, then averaged per week.

4. **Special Sales → Weekly:** Binary flag — if *any* day in the week had a special sale, the whole week is flagged `1`.

### 💬 Discussion Point for Audience

> *"Why not just use the 12 monthly rows? What breaks?"*  
> Answer: With 12 observations and 10+ features, OLS has more unknowns than equations. The model will overfit perfectly (R²=1.0) but learn nothing generalisable. It's mathematically underdetermined.

---

## 3. Feature Engineering: Adstock & Saturation (10 min)

📄 **Script:** [`Model/feature_engineering_mmm.py`](../Model/feature_engineering_mmm.py)

### Concept 1: Adstock (The "Memory" Effect)

A TV ad shown this week still influences purchase decisions next week, and the week after — with diminishing impact. We model this as **geometric decay**:

```
Adstock(t) = Spend(t) + α × Adstock(t-1)
```

Where `α` (alpha) is the decay rate:

| Channel | α (Decay) | Interpretation |
|---|---|---|
| TV | 0.7 | Long memory — brand recall lasts weeks |
| Radio | 0.5 | Medium memory |
| SEM | 0.1 | Very short — you search, you click, you buy |
| Affiliates | 0.2 | Transactional, little carry-over |

```python
def apply_adstock(series, alpha):
    adstocked = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            adstocked[i] = series[i]
        else:
            adstocked[i] = series[i] + alpha * adstocked[i - 1]
    return adstocked
```

### Concept 2: Diminishing Returns (The "Saturation" Effect)

Spending $10K might get 5,000 clicks, but $100K won't get 50,000. We apply a **log transform** to flatten the curve:

```python
saturated_adstock = np.log1p(adstocked_spend)  # log(x + 1)
```

### 🖼️ Visual Explanation

```
Raw Spend          →    After Adstock       →    After Log Transform
                        (carries forward)        (flattens at high spend)

 $                       $                        $
 │  ╷                    │    ╱─                  │      ────────
 │  │                    │   ╱                    │    ╱
 │  │                    │  ╱                     │  ╱
 │  │                    │ ╱                      │╱
 └──┴──── weeks          └────── weeks            └────── weeks
```

### 💬 Discussion Point

> *"In a production model (e.g., Google Meridian or Meta Robyn), the alpha values aren't assumed — they're learned from the data. How would you let the model learn them?"*  
> Answer: You parameterise α as a Beta(2,2) prior in the Bayesian model, giving each channel its own learnable decay rate.

---

## 4. The OLS Baseline — And Why It Fails (10 min)

📄 **Script:** [`Model/train_baseline_mmm.py`](../Model/train_baseline_mmm.py)

### The Setup

```python
# Standard linear regression: GMV = β₀ + β₁·TV + β₂·Digital + ... + ε
model = sm.OLS(y, X_scaled).fit()
```

### The Problem: Negative Coefficients

| Feature | Coefficient |
|---|---|
| Online Marketing | +$66.2M ✅ |
| Special_Sale_Flag | +$23.1M ✅ |
| **Content Marketing** | **-$24.5M ❌** |
| **Affiliates** | **-$55.8M ❌** |

The model says *"spending on Content Marketing reduces sales by $24.5M"* — which is economically nonsensical.

### Root Cause: Multicollinearity

When brands run campaigns, they often spend on **multiple channels simultaneously** (TV + Digital + Content all spike during holiday periods). OLS can't untangle the individual effects and arbitrarily assigns negative weights to break the tie.

```
             ┌── TV Spend ──────┐
Holiday ─────┤── Digital Spend ──├──→ Sales Spike
Season       └── Content Spend ─┘

OLS sees all 3 correlated with sales, can't separate them,
and randomly assigns negative weight to some.
```

### 💬 Discussion Point

> *"The OLS R² (0.57) is actually higher than the Bayesian R² (0.47). Does that mean OLS is 'better'?"*  
> Answer: No! OLS is overfitting by using negative coefficients as a mathematical trick. The Bayesian model trades some R² for **economically valid** results. R² isn't everything.

---

## 5. The Bayesian Solution — PyMC (15 min)

📄 **Script:** [`Model/train_bayesian_mmm.py`](../Model/train_bayesian_mmm.py)

### The Core Idea: Priors as Business Rules

Unlike OLS which looks at data blankly, Bayesian models let us **inject domain knowledge** as mathematical constraints:

```python
with pm.Model() as mmm:
    # Rule 1: Base sales can be anything
    intercept = pm.Normal("intercept", mu=0, sigma=1)
    
    # Rule 2: Special Sales / NPS can be positive or negative
    beta_base = pm.Normal("beta_base", mu=0, sigma=1, shape=2)
    
    # Rule 3: ⭐ MARKETING CAN NEVER BE NEGATIVE ⭐
    beta_media = pm.HalfNormal("beta_media", sigma=1, shape=9)
    
    # The prediction equation (same as OLS, different constraints)
    mu = intercept + X_base @ beta_base + X_media @ beta_media
    
    # Train via MCMC (Markov Chain Monte Carlo)
    trace = pm.sample(draws=1000, tune=1000, chains=2)
```

### What is `HalfNormal`?

```
Normal Distribution        HalfNormal Distribution
(OLS allows this)          (Bayesian constrains to this)

      ╱╲                          │╲
    ╱    ╲                        │  ╲
  ╱        ╲                      │    ╲
╱────────────╲                    │──────╲
-3  -1  0  1  3                   0   1   2   3
     ↑                                ↑
 Negative weights                 Only positive
 are allowed!                     weights allowed!
```

### The Results: All Positive Weights

| Channel | Bayesian Weight | HDI 3% | HDI 97% |
|---|---|---|---|
| Special_Sale_Flag | 0.358 | 0.158 | 0.572 |
| **Sponsorship** | **0.300** | 0.003 | 0.612 |
| Online Marketing | 0.146 | 0.000 | 0.382 |
| Affiliates | 0.121 | 0.000 | 0.325 |
| SEM | 0.117 | 0.000 | 0.281 |
| Radio | 0.101 | 0.000 | 0.240 |
| Digital | 0.093 | 0.000 | 0.237 |
| TV | 0.088 | 0.000 | 0.233 |
| Content Marketing | 0.080 | 0.000 | 0.206 |

**R² = 0.47** — lower than OLS, but every coefficient is economically valid.

### What is HDI (Highest Density Interval)?

The HDI is like a confidence interval but better — it tells us the range of values that are **94% likely** given the data.

- Sponsorship HDI: [0.003, 0.612] → we're confident it's positive, but there's wide uncertainty on *how much*
- This uncertainty is **valuable** — it tells the PM which channels need more experimental validation

### 💬 Discussion Point

> *"Why use MCMC sampling instead of just constraining OLS with bounds?"*  
> Answer: Bounded OLS (e.g., `scipy.optimize.minimize` with bounds) finds a point estimate. MCMC gives you a **full posterior distribution** — you get uncertainty estimates for free, which is critical for budget decisions.

---

## 6. ROAS Evaluation & Calibration (10 min)

📄 **Script:** [`Model/evaluate_mmm.py`](../Model/evaluate_mmm.py)

### Step 1: Calculate ROAS from Model Weights

```python
# Total media-driven GMV (heuristic: 70% of explained variance)
media_driven_gmv = total_gmv * (r_squared * 0.7)

# Split by channel based on relative importance
channel_sales = share_of_contribution * media_driven_gmv

# ROAS = attributed sales / actual spend
roas = channel_sales / total_spend
```

### Results: Channel Rankings

| Channel | Total Spend | Sales Driven | ROAS |
|---|---|---|---|
| 🥇 **Radio** | $4.7K | $13.0M | **$2.76M** |
| 🥈 **TV** | $44.4K | $94.5M | **$2.13M** |
| 🥉 **Sponsorship** | $365.4K | $765.5M | **$2.09M** |
| Online Marketing | $193.7K | $217.8M | $1.12M |
| Other | $48.0K | $36.9M | $0.77M |

### Step 2: Why These Numbers Need Calibration

**Critical insight:** These ROAS numbers are **correlation upper-bounds**, not causal truths.

### The Three Lenses of Measurement

```
                    ┌────────────────────────────────────┐
                    │    Sponsorship: 3 Different Answers │
                    ├────────────────────────────────────┤
                    │                                    │
  DDA (Last-Click)  │  ROAS = $0.3M   ← UNDERSTATES     │
                    │  (Can't track offline impressions)  │
                    │                                    │
  MMM (This Model)  │  ROAS = $2.1M   ← OVERSTATES      │
                    │  (Correlation with holiday season)  │
                    │                                    │
  Incrementality    │  ROAS = $0.9M   ← GROUND TRUTH     │
  Experiment        │  (Geo-holdout proves causation)    │
                    └────────────────────────────────────┘
```

### The Calibration Multiplier

```python
# From geo-holdout experiment:
true_roas = 0.9   # Measured by turning off Sponsorship in Texas
model_roas = 2.1  # What our model says

calibration_multiplier = true_roas / model_roas  # = 0.428

# Apply to all model outputs:
calibrated_sales = modelled_sales * 0.428
```

| Metric | Raw MMM Output | After Calibration |
|---|---|---|
| Sales Contribution | $765M | **$327M** |
| ROAS | $2.1M | **$0.9M** |

### 💬 Discussion Point

> *"How often should you re-calibrate?"*  
> Answer: Google recommends running incrementality experiments for each major channel at least once every 6 months. The calibration multiplier drifts as market conditions change.

---

## 7. Hands-On Exercises (10 min)

### Exercise 1: Change the Adstock Decay Rate (Easy — 5 min)

In `feature_engineering_mmm.py`, change TV's alpha from `0.7` to `0.3` and re-run the full pipeline (Steps 2–5).

**Questions to answer:**
- How does TV's Bayesian weight change?
- Does Radio's ROAS ranking change?
- What does this tell you about sensitivity to assumed alphas?

### Exercise 2: Add a New Feature (Medium — 10 min)

Add a **trend variable** (week number 1–52) to capture organic growth over the year.

```python
# In feature_engineering_mmm.py, add:
engineered_df["Week_Number"] = range(1, len(engineered_df) + 1)
```

Then add `"Week_Number"` to `base_features` in both training scripts.

**Questions to answer:**
- Does R² improve?
- Do the media channel weights change? Why or why not?

### Exercise 3: Production Architecture (Discussion — 10 min)

Design a system architecture diagram for a production MMM that:
1. Ingests weekly data from Snowflake
2. Runs the Bayesian model on a schedule
3. Serves calibrated ROAS to a Streamlit dashboard
4. Lets the CMO run "what-if" budget scenarios

---

## 📚 Further Reading

| Resource | Why Read It |
|---|---|
| [Google Meridian (Open-Source)](https://github.com/google/meridian) | Production-grade Bayesian MMM in JAX |
| [Meta Robyn (Open-Source)](https://github.com/facebookexperimental/Robyn) | Ridge regression MMM with automatic hyperparameter tuning |
| [PyMC Marketing](https://www.pymc-marketing.io/) | PyMC's official MMM/CLV library |
| [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) | Best intro to Bayesian thinking |
| [Google's Measurement Playbook](https://www.thinkwithgoogle.com/) | Strategic framework for dual-speed measurement |

---

## 🔑 Key Takeaways (Slide for Wrap-Up)

1. **MMM is a statistical model, not a tracking tool** — it works at the aggregate level and doesn't need cookies
2. **OLS fails for marketing data** — multicollinearity makes coefficients untrustworthy
3. **Bayesian priors = domain knowledge in code** — `HalfNormal` is how you tell the model "marketing can't hurt sales"
4. **ROAS from MMM is an upper bound** — always calibrate with incrementality experiments
5. **No single measurement tool is complete** — use the Dual-Speed Loop (MMM for strategy, DDA for tactics, Experiments for truth)

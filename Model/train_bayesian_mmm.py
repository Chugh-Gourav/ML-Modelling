import pandas as pd
import numpy as np
import os
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# ==============================================================================
# AI PRODUCT MANAGER'S GUIDE TO BAYESIAN MMM (using PyMC)
# ==============================================================================
# Bayesian models are the gold standard for MMM (used by Google & Meta).
#
# Why? The concept of "PRIORS".
# Unlike Linear Regression which looks at the data blankly and says "Affiliate marketing
# caused a $50M loss in sales", Bayesian models allow us to inject HUMAN DOMAIN KNOWLEDGE.
#
# We tell the AI: "Here is my data, BUT I want you to strictly follow these rules (Priors):"
# 1. Marketing CANNOT have a negative impact. Stop trying to make coefficients negative.
#    (We enforce this using a Half-Normal distribution which only allows positive numbers >= 0).
# 2. Base sales (what we make if we turn off all ads) must be positive.
#
# By forcing the math to respect these business realities, the Bayesian model gracefully
# handles multicollinearity (channels inflating each other) and gives us true,
# actionable ROAS (Return on Ad Spend) readouts that we can take to the CFO.
# ==============================================================================

print("Loading Engineered Data...")
df = pd.read_csv(os.path.join(data_dir, "Engineered_MMM_Data.csv"))

target = "GMV"
base_features = ["Special_Sale_Flag", "NPS"]
media_features = [col for col in df.columns if "_Adstock_Log" in col]

X_base = df[base_features].values
X_media = df[media_features].values
y = df[target].values

# 1. PRE-PROCESSING
# Bayesian models take a long time to sample (train). Scaling the data
# significantly speeds up the math engine (NUTS Sampler).
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

scaler_base = StandardScaler()
X_base_scaled = scaler_base.fit_transform(X_base)

scaler_media = StandardScaler()
X_media_scaled = scaler_media.fit_transform(X_media)

# 2. BUILD THE BAYESIAN MODEL
print("\nBuilding the Bayesian PyMC Model...")
with pm.Model() as mmm:

    # --- PRIORS (The Business Rules) ---
    # Intercept: Base sales if we do nothing. Normal distribution around 0 (since y is scaled to mean 0)
    intercept = pm.Normal("intercept", mu=0, sigma=1)

    # Base Coefficients: Special Sales / NPS could theoretically be negative or positive,
    # so we use a standard Normal distribution.
    beta_base = pm.Normal("beta_base", mu=0, sigma=1, shape=X_base.shape[1])

    # MEDIA COEFFICIENTS: This is the magic. HalfNormal forces all weights to be >= 0.
    # The model CANNOT say that TV spend hurt sales.
    beta_media = pm.HalfNormal("beta_media", sigma=1, shape=X_media.shape[1])

    # Model Error (Noise)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # --- LIKELIHOOD (The Math Equation) ---
    # Prediction = Intercept + (Base_Features * Base_Weights) + (Media_Features * Media_Weights)
    mu = (
        intercept
        + pm.math.dot(X_base_scaled, beta_base)
        + pm.math.dot(X_media_scaled, beta_media)
    )

    # Compare prediction to actual Y
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_scaled)

    # 3. TRAIN THE MODEL (MCMC Sampling)
    print("\nTraining the model via MCMC Sampling (This may take 1-2 minutes)...")
    # Using 1000 draws to balance speed and accuracy for this baseline
    trace = pm.sample(
        draws=1000, tune=1000, chains=2, cores=1, random_seed=42, progressbar=False
    )

# 4. EXTRACT RESULTS & EVALUATE
print("\nExtracting Results...")
# Get the "mean" weight the AI settled on for each feature across the 1000 draws
summary = pm.summary(trace, var_names=["intercept", "beta_base", "beta_media"])

# Let's map the generic names back to our actual channel names
feature_names = ["Base_Intercept"] + base_features + media_features
# Grab only the 'mean' column from the PyMC summary
weights = summary["mean"].values

# Create clean dataframe
coeff_df = pd.DataFrame({
    "Feature": feature_names, 
    "Bayesian_Weight": weights,
    "HDI_Lower_3%": summary["hdi_3%"].values,
    "HDI_Upper_97%": summary["hdi_97%"].values
})
coeff_df = coeff_df.sort_values(by="Bayesian_Weight", ascending=False).reset_index(
    drop=True
)

print("\n==============================================")
print("BAYESIAN CHANNEL COEFFICIENTS (STRICTLY POSITIVE)")
print("==============================================")
print(coeff_df.to_string())

# Calculate R-Squared (by simulating predictions using the learned mean weights)
y_pred_scaled = (
    weights[0]
    + np.dot(X_base_scaled, weights[1:3])
    + np.dot(X_media_scaled, weights[3:])
)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

r2 = r2_score(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

print("\n==============================================")
print(f"BAYESIAN MODEL FIT EVALUATION")
print("==============================================")
print(f"R-squared: {r2:.4f}")
print(f"MAPE: {mape*100:.1f}%")
print("==============================================\n")

# Save results to a Text File
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Insights", "Bayesian_MMM_Results.txt")
with open(results_path, "w") as f:
    f.write("BAYESIAN MMM RESULTS\n\n")
    f.write(coeff_df.to_string())
    f.write(f"\n\nR-squared: {r2:.4f}")
print(f"Results saved to {results_path}")

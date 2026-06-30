import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# ==============================================================================
# AI PRODUCT MANAGER'S GUIDE TO THE BASELINE LINEAR REGRESSION MODEL
# ==============================================================================
# Why start with a simple Linear Regression before jumping into fancy AI or Bayesian models?
# 1. Interpretability: We can clearly see the "coefficient" (weight/importance) of every channel.
# 2. Baseline to beat: If complex Bayesian models can't beat this simple math, they aren't worth the cost.
# 3. Warning on Multicollinearity: In the real world, brands spend on TV, Search, and Display
#    all at the exact same time. This confuses simple linear models because the variables are
#    highly correlated. We must watch out for negative coefficients (which would falsely imply
#    spending *more* on ads causes sales to *drop*).
# ==============================================================================

print("Loading Engineered Data...")
df = pd.read_csv(os.path.join(data_dir, "Engineered_MMM_Data.csv"))

# 1. DEFINE OUR VARIABLES
# The target we want to predict is our Weekly Sales (GMV)
target = "GMV"

# The features we will use to predict it: Standard features + our engineered Adstock Log channels
base_features = ["Special_Sale_Flag", "NPS"]
media_features = [col for col in df.columns if "_Adstock_Log" in col]
all_features = base_features + media_features

X = df[all_features].copy()
y = df[target].copy()

# 2. STANDARDIZATION
# It is best practice to scale the features so they all live on the same numerical scale (e.g., mean 0, std 1).
# This makes it easier for the model to compare the true weight of "NPS" vs "TV Spend".
print("\nStandardizing Features...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 3. ADD CONSTANT (INTERCEPT)
# We must manually add an intercept to our data for the statsmodels OLS formula.
# In marketing terms, the intercept is the "Base Sales" you would have gotten if you
# spent $0 on marketing and had a 0 NPS.
X_scaled = sm.add_constant(X_scaled)

# 4. FIT THE MODEL
print("\nFitting Ordinary Least Squares (OLS) Linear Regression...")
model = sm.OLS(y, X_scaled).fit()

# 5. PREDICT AND EVALUATE
y_pred = model.predict(X_scaled)

r2 = r2_score(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

print("\n==============================================")
print(f"MODEL FIT EVALUATION")
print("==============================================")
print(
    f"R-squared: {r2:.4f} (Model explains {r2*100:.1f}% of the weekly sales variance)"
)
print(f"MAPE: {mape*100:.1f}% (On average, predictions are off by this percentage)")
print("==============================================\n")

print("==============================================")
print("CHANNEL COEFFICIENTS (IMPACT WEIGHTS)")
print("==============================================")
# We extract the coefficients. A positive number means it drives sales up.
# A negative number is usually a sign of multicollinearity breaking the linear model.
coeff_df = pd.DataFrame({"Feature": X_scaled.columns, "Coefficient": model.params})
coeff_df = coeff_df.sort_values(by="Coefficient", ascending=False).reset_index(
    drop=True
)
print(coeff_df.to_string())

# Save results to review later
results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Insights", "OLS_Regression_Results.txt")
with open(results_path, "w") as f:
    f.write(model.summary().as_text())
print(f"\nFull detailed statistical summary saved to {results_path}")

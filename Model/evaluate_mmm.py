import pandas as pd
import numpy as np

# 1. Load the Engineered Dataset and Model Results
df = pd.read_csv("../Data/Engineered_MMM_Data.csv")
results_df = pd.read_csv(
    "../Insights/Bayesian_MMM_Results.txt",
    sep="\s{2,}",
    skiprows=2,
    nrows=12,
    engine="python",
)
results_df.columns = ["Feature", "Bayesian_Weight"]

# Extract R-squared value
with open("../Insights/Bayesian_MMM_Results.txt", "r") as f:
    lines = f.readlines()
    r_squared_line = [line for line in lines if "R-squared:" in line][0]
    r_squared = float(r_squared_line.split(":")[1].strip())


# 2. Extract specific media columns to calculate contributions
media_features = [
    "TV_Adstock_Log",
    "Digital_Adstock_Log",
    "Sponsorship_Adstock_Log",
    "Content Marketing_Adstock_Log",
    "Online marketing_Adstock_Log",
    " Affiliates_Adstock_Log",
    "SEM_Adstock_Log",
    "Radio_Adstock_Log",
    "Other_Adstock_Log",
]

# We need the original spend columns to calculate ROAS (Return vs Spend)
original_spend_columns = [
    "TV",
    "Digital",
    "Sponsorship",
    "Content Marketing",
    "Online marketing",
    " Affiliates",
    "SEM",
    "Radio",
    "Other",
]


# 3. Calculate Overall Contributions
# For each week, the contribution of a channel = its standardized value * its bayesian weight.
# To keep it simple and interpretable for the PM, we will calculate the relative % contribution to the predicted variance.

# Get the standard deviation of each feature from the standardized data (used during training)
# (Since we standardized the data before training, the weights belong to the standardized scale)
# Let's approximate the relative importance by multiplying the mean of the transformed feature by the weight
feature_means = df[media_features].mean()

contributions = []
for index, row in results_df.iterrows():
    feature = row["Feature"]
    weight = row["Bayesian_Weight"]

    if feature in media_features:
        mean_val = feature_means[feature]
        # Approximation of average contribution
        avg_contribution = mean_val * weight
        contributions.append(
            {
                "Channel": feature.replace("_Adstock_Log", ""),
                "Relative_Importance": avg_contribution,
                "Weight": weight,
            }
        )

contrib_df = pd.DataFrame(contributions)

# Normalize to 100% to show share of media contribution
contrib_df["Share_of_Media_Contribution_%"] = (
    contrib_df["Relative_Importance"] / contrib_df["Relative_Importance"].sum()
) * 100

# 4. Calculate ROAS (Return on Ad Spend)
# ROAS = (Sales driven by Channel) / (Spend on Channel)
# Note: In a rigorous MMM, we 'un-standardize' the predictions to get exact dollar values.
# For this prototype, we'll assign the total historical GMV proportionally based on the 'Share_of_Media_Contribution',
# and compare it to the actual total spend per channel.
# (Assumption: Base sales + NPS + Special Sales account for a portion, Media accounts for the rest.
# For simplicity in this demo, let's look at the *relative* efficiency).

total_gmv = df["GMV"].sum()
# Assume media drives roughly 30% of total sales (Typical industry benchmark, as the R-sq is ~47%, media + base = 47% variance explained)
# Let's just distribute the "media-driven" portion to get a ROAS multiplier
media_driven_gmv = total_gmv * (
    r_squared * 0.7
)  # Heuristic: 70% of the explained variance is media

contrib_df["Estimated_Sales_Driven"] = (
    contrib_df["Share_of_Media_Contribution_%"] / 100
) * media_driven_gmv

base_df = pd.read_csv("../Data/Weekly_MMM_Data.csv")

# Calculate Total Spend per channel
spend_totals = []
for col in original_spend_columns:
    spend_totals.append({"Channel": col.strip(), "Total_Spend": base_df[col].sum()})
spend_df = pd.DataFrame(spend_totals)

# Merge
roas_df = pd.merge(contrib_df, spend_df, on="Channel", how="left")

# Calculate ROAS
roas_df["ROAS"] = roas_df["Estimated_Sales_Driven"] / roas_df["Total_Spend"]

# Sort by ROAS
roas_df = roas_df.sort_values(by="ROAS", ascending=False)

print("\n--- Media Mix Optimization Insights ---")
print(f"Model R-Squared: {r_squared:.4f}")
print("\n1. Channel Weights (Positive constraint enforced):")
print(
    roas_df[["Channel", "Weight", "Share_of_Media_Contribution_%"]].to_string(
        index=False
    )
)

print("\n2. Estimated Return on Ad Spend (ROAS):")
print(
    roas_df[["Channel", "Total_Spend", "Estimated_Sales_Driven", "ROAS"]].to_string(
        index=False
    )
)

roas_df.to_csv("../Insights/MMM_ROAS_Insights.csv", index=False)
print("\nSaved detailed insights to 'MMM_ROAS_Insights.csv'")

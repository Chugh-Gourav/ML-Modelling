import pandas as pd
import numpy as np
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# ==============================================================================
# AI PRODUCT MANAGER'S GUIDE TO FEATURE ENGINEERING FOR MMM
# ==============================================================================
# In a Media Mix Model (MMM), marketing spend doesn't just impact sales on the exact
# day or week the money was spent. Marketing has two crucial psychological effects
# that we must mathematically model before feeing the data to our Machine Learning algorithms:
#
# 1. ADSTOCK (The "Memory" or "Carry-over" Effect)
#    - What it is: If you see a TV ad today, you might not buy the product until next week.
#      The impact of that ad "decays" over time.
#    - How we model it: We use a decay rate (alpha, between 0 and 1).
#      Formula: Adstock_Value(Week T) = Spend(Week T) + (alpha * Adstock_Value(Week T-1))
#      For example, if alpha = 0.5, then 50% of last week's ad power carries over to this week.
#    - AI Product PM Insight: Different channels have different 'memories'.
#      TV ads usually have a high alpha (long memory, maybe 0.7) because they build brand awareness.
#      Search Engine Marketing (SEM) has a low alpha (short memory, maybe 0.1) because it's meant for immediate conversion.
#
# 2. DIMINISHING RETURNS (The "Saturation" Effect)
#    - What it is: Spending $10,000 might get you 5,000 clicks. But spending $100,000 probably
#      won't get you 50,000 clicks. People get tired of seeing the ad (ad fatigue), or you run
#      out of interested people to show it to.
#    - How we model it: We apply a non-linear transformation like a Logarithm (log(x)) or a Root curve.
#      This makes the curve bend flat after a certain point of spend.
#    - AI Product PM Insight: Identifying exactly where this curve flattens out helps you
#      tell your marketing team "Stop spending on Digital, we've saturated the audience,
#      move the budget to TV instead."
# ==============================================================================


def apply_adstock(series, alpha):
    """
    Applies the Adstock geometric decay to a pandas series safely.
    """
    adstocked = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            adstocked[i] = series[i]
        else:
            adstocked[i] = series[i] + alpha * adstocked[i - 1]
    return adstocked


print("Loading Weekly Data...")
weekly_df = pd.read_csv(os.path.join(data_dir, "Weekly_MMM_Data.csv"))

# Let's define the columns that represent our marketing channels
media_channels = [
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

# For this baseline, we will apply a set of assumed "alphas" (decay rates).
# Note: In our advanced Bayesian Model (Step 4), the AI will actually "learn" these optimal
# alphas by itself from the data! But for our linear baseline, we assume them based on industry standards.
assumed_alphas = {
    "TV": 0.7,  # Highly memorable offline channel (Long decay)
    "Radio": 0.5,  # Memorable but less so than video (Medium decay)
    "Sponsorship": 0.6,  # Builds brand equity over time
    "Content Marketing": 0.4,  # Blogs/videos have some lingering value
    "Digital": 0.3,  # Fast-moving channel (Short decay)
    "Online marketing": 0.3,  # Fast-moving channel
    " Affiliates": 0.2,  # Transactional channel
    "SEM": 0.1,  # Very immediate, you search -> you click -> you buy (Very short decay)
    "Other": 0.3,  # Generic
}

print("Applying Adstock and Diminishing Returns...")
# Create a new DataFrame to hold our engineered features alongside our target variable (GMV)
engineered_df = weekly_df[
    ["Date", "GMV", "Total Investment", "Special_Sale_Flag", "NPS"]
].copy()

for channel in media_channels:
    if channel in weekly_df.columns:
        alpha = assumed_alphas.get(channel, 0.5)  # Default to 0.5 if not found

        # 1. Apply Adstock (Carry-over)
        adstocked_spend = apply_adstock(weekly_df[channel].values, alpha)

        # 2. Apply Diminishing Returns (Saturation)
        # We use a natural log log(x + 1). The '+1' is a mathematical trick to prevent log(0) which is undefined.
        saturated_adstock = np.log1p(adstocked_spend)

        # Save the new engineered feature to our fresh dataframe
        feature_name = f"{channel}_Adstock_Log"
        engineered_df[feature_name] = saturated_adstock

# Save the transformed dataset
output_path = os.path.join(data_dir, "Engineered_MMM_Data.csv")
engineered_df.to_csv(output_path, index=False)
print(f"Engineered dataset saved successfully to: {output_path}")

# Let's show the user how a single channel (like TV) changed over the first 5 weeks
print("\n--- Example: TV Transformation (First 5 Weeks) ---")
demo_df = pd.DataFrame(
    {
        "Raw_TV_Spend": weekly_df["TV"].head().values,
        "Adstocked_TV (70% carryover)": apply_adstock(
            weekly_df["TV"].head().values, 0.7
        ),
        "Final_TV_Feature (Log Saturation)": engineered_df["TV_Adstock_Log"]
        .head()
        .values,
    }
)
print(demo_df.to_string())

import pandas as pd
import numpy as np
import os

# --- CONTEXT ---
# This script converts our raw data (which comes in a mix of Daily and Monthly formats)
# into a uniform "Weekly" dataset. Why weekly? Because Media Mix Models (MMM) need a
# granular time series to properly measure "Adstock" (the delayed effect of advertising).
# Monthly data (only 12 rows) is too small, so we break it down to 52-53 weeks.

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# ==============================================================================
# 1. LOAD & AGGREGATE SALES TO WEEKLY LEVEL
# ==============================================================================
print("Loading Sales Data...")
# Load the 150MB granular daily sales file. We only need Date and GMV (Revenue)
sales_df = pd.read_csv(
    os.path.join(data_dir, "Raw Data", "firstfile.csv"),
    usecols=["Date", "gmv_new"],
    parse_dates=["Date"],
)

# 'Grouper' groups the daily dates into weeks ending on Sunday ('W-SUN').
# We sum the GMV for all days in that week.
weekly_sales = (
    sales_df.groupby(pd.Grouper(key="Date", freq="W-SUN"))["gmv_new"]
    .sum()
    .reset_index()
)
weekly_sales.rename(columns={"gmv_new": "GMV"}, inplace=True)
print(f"Weekly Sales shape: {weekly_sales.shape}")

# ==============================================================================
# 2. LOAD & INTERPOLATE MEDIA INVESTMENT TO WEEKLY LEVEL
# ==============================================================================
print("Loading Media Investment...")
media_df = pd.read_csv(os.path.join(data_dir, "Raw Data", "MediaInvestment.csv"))
media_df.fillna(0, inplace=True)  # Replace missing values with 0

# Create a 'Month_Start' date column from the Year and Month columns
media_df["Month_Start"] = pd.to_datetime(
    media_df["Year"].astype(str) + "-" + media_df["Month"].astype(str) + "-01"
)

# Since media spend is Monthly, we need to divide it smoothly into Weekly.
# To do this accurately, we first break it down into Daily spend, then re-aggregate to Weekly.
daily_dates = pd.date_range(start="2015-07-01", end="2016-06-30", freq="D")
daily_df = pd.DataFrame({"Date": daily_dates})
daily_df["Month_Start"] = daily_df["Date"].dt.to_period("M").dt.to_timestamp()
daily_df["Days_in_Month"] = daily_df["Date"].dt.days_in_month

# Merge the monthly spend values onto every single day of that month
daily_media = pd.merge(daily_df, media_df, on="Month_Start", how="left")

# Divide monthly spend by the number of days in that month to get "Daily Spend"
media_cols = [
    "Total Investment",
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
for col in media_cols:
    if col in daily_media.columns:
        daily_media[col] = pd.to_numeric(daily_media[col], errors="coerce").fillna(0)
        daily_media[col] = daily_media[col] / daily_media["Days_in_Month"]

# Now re-aggregate those daily spend numbers up to Weekly (summing the days in each week)
weekly_media = (
    daily_media.groupby(pd.Grouper(key="Date", freq="W-SUN"))[media_cols]
    .sum()
    .reset_index()
)
print(f"Weekly Media shape: {weekly_media.shape}")

# ==============================================================================
# 3. LOAD & AGGREGATE SPECIAL SALES (HOLIDAYS/EVENTS)
# ==============================================================================
print("Loading Special Sales...")
special_sales = pd.read_csv(os.path.join(data_dir, "Raw Data", "SpecialSale.csv"))
special_sales["Date"] = pd.to_datetime(special_sales["Date"])
special_sales["Special_Sale_Flag"] = 1  # Mark these specific dates with a 1

# If any day in a given week was a special sale, we flag the entire week (max() logic)
weekly_special = (
    special_sales.groupby(pd.Grouper(key="Date", freq="W-SUN"))["Special_Sale_Flag"]
    .max()
    .reset_index()
)
weekly_special["Special_Sale_Flag"] = weekly_special["Special_Sale_Flag"].fillna(0)

# ==============================================================================
# 4. LOAD & AGGREGATE NPS (Net Promoter Score)
# ==============================================================================
print("Loading NPS...")
nps_df = pd.read_csv(
    os.path.join(data_dir, "Raw Data", "MonthlyNPSscore.csv"), parse_dates=["Date"]
)

# Similar to media, NPS is monthly. We forward-fill it to get it to the daily level,
# then average it across the week.
daily_nps = pd.DataFrame({"Date": daily_dates})
daily_nps = pd.merge(daily_nps, nps_df, on="Date", how="left")
daily_nps["NPS"] = daily_nps["NPS"].ffill().bfill()
weekly_nps = (
    daily_nps.groupby(pd.Grouper(key="Date", freq="W-SUN"))["NPS"].mean().reset_index()
)

# ==============================================================================
# 5. MERGE ALL WEEKLY COMPONENTS INTO ONE FINAL DATASET
# ==============================================================================
print("Merging data...")
# Inner merge Sales and Media on the exact week
final_df = pd.merge(weekly_sales, weekly_media, on="Date", how="inner")
# Left merge special sales and NPS
final_df = pd.merge(final_df, weekly_special, on="Date", how="left")
final_df["Special_Sale_Flag"] = final_df["Special_Sale_Flag"].fillna(0)
final_df = pd.merge(final_df, weekly_nps, on="Date", how="left")

# Save the final cleaned dataframe to CSV
output_path = os.path.join(data_dir, "Weekly_MMM_Data.csv")
final_df.to_csv(output_path, index=False)
print(f"Final Data shape: {final_df.shape}")
print(f"Successfully saved to {output_path}\n")

print("--- SAMPLE OF FIRST 10 WEEKS ---")
print(final_df.head(10).to_string())

import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# Load the weekly data
df = pd.read_csv(os.path.join(data_dir, 'Weekly_MMM_Data.csv'))

# Export the first 10 rows to Excel as requested
output_sample_path = os.path.join(data_dir, 'Weekly_MMM_Data_Sample_10_Rows.xlsx')
df.head(10).to_excel(output_sample_path, index=False)

# Also export the full dataset to Excel for completeness
output_full_path = os.path.join(data_dir, 'Weekly_MMM_Data_Full.xlsx')
df.to_excel(output_full_path, index=False)

print(f"Sample exported to: {output_sample_path}")
print(f"Full dataset exported to: {output_full_path}")

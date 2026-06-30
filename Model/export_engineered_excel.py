import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data")

# Load the engineered weekly data
df = pd.read_csv(os.path.join(data_dir, 'Engineered_MMM_Data.csv'))

# Export the full dataset to Excel 
output_full_path = os.path.join(data_dir, 'Engineered_MMM_Data_Full.xlsx')
df.to_excel(output_full_path, index=False)

print(f"Engineered dataset exported to Excel: {output_full_path}")

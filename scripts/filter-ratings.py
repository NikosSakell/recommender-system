import pandas as pd
from datetime import datetime

# Read the CSV file
file_path = 'ratings.csv'
df = pd.read_csv(file_path)

# Convert timestamp to a recognizable time format
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Apply data reduction algorithm
# In this example, we'll drop every 5 rows
df_reduced = df.iloc[::5]

# Save the cleaned and reduced data to a new CSV file
output_path = 'ratings-reduced.csv'
df_reduced.to_csv(output_path, index=False)

print(f"Original file size: {df.shape}")
print(f"Reduced file size: {df_reduced.shape}")

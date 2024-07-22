import pandas as pd

# Load your CSV file into a Pandas DataFrame.
df = pd.read_csv('combined_runs.csv')

# Increment all values in column 'column_name' by 1.
df['epoch'] += 1
df['issue'] += 1
df['step'] += 1

# Save the modified DataFrame back to another CSV file.
df.to_csv('allruns.csv', index=False)

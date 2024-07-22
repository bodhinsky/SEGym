import pandas as pd
import pyarrow as pa
from pyarrow.parquet import write_table

# Load CSV into DataFrame
df = pd.read_csv('combined_runs.csv')

table = pa.Table.from_pandas(df)

# Convert DataFrame to Parquet format
parquet_file = 'output.parquet'
write_table(table, parquet_file)

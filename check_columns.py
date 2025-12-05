import pandas as pd
try:
    df = pd.read_parquet('/home/admin/proj_LAIA/data/raw/yellow_tripdata_2011-01.parquet')
    print(list(df.columns))
except Exception as e:
    print(e)

### download numerai live data

# load libraries
import pandas as pd
from numerapi import NumerAPI

# instantiate numerai client
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
current_round = napi.get_current_round()
print(f"Current round #: {current_round}")

# donwload live dataset
napi.download_dataset(
    filename="v4/live.parquet", 
    dest_path=f"data/live_{current_round}.parquet"
)
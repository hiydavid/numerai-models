# download numerai live data

# load libraries
import pandas as pd
from numerapi import NumerAPI
import os
from dotenv import load_dotenv 

# instantiate env var
load_dotenv()
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')

# instantiate numerai client
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
current_round = napi.get_current_round()
print(f"Current round #: {current_round}")

# donwload live dataset
napi.download_dataset(
    filename="v4/live.parquet", 
    dest_path=f"data/live_{current_round}.parquet"
)
# download numerai live data

# print
print("DAILY PROCESS BEGIN!")

# load libraries
import os
import gc
import json
from numerapi import NumerAPI
from utils.models import RunModel

# instantiate env var
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

# init class
nmr = RunModel(roundn=current_round, mode="live")
nmr.get_data()

# run foxhound
nmr.run_foxhound()
gc.collect()

# run deadcell
nmr.run_deadcell()
gc.collect()

# run cobra
nmr.run_cobra()
gc.collect()

# run beautybeast
nmr.run_beautybeast()
gc.collect()

# run skulls
nmr.run_skulls()
gc.collect()

# run desperado
nmr.run_desperadov3()
gc.collect()

# run gaia
nmr.run_gaia()
gc.collect()

# run terra
nmr.run_terra()
gc.collect()

# read model name json file
with open("data/model_names.json", "r") as f:
    model_names = json.load(f)

# submit live predictions for current round
for item in model_names.items():
    print(f"Submitting live predictions for {item[0]}...")
    napi.upload_predictions(
        file_path=f"predictions/{item[0]}_live_preds_{current_round}.csv",
        model_id=item[1]
    )

# print
print("DAILY PROCESS COMPLETE!")
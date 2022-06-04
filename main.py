# load libraries
import pandas as pd
import json
from numerapi import NumerAPI
from utils.api_keys import PUBLIC_ID, SECRET_KEY
from utils.run_model import RunModel

# instantiate new round
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
current_round = napi.get_current_round()
print(f"Starting round #: {current_round}")

# donwload live dataset
napi.download_dataset(
    filename="v4/live.parquet", 
    dest_path=f"data/live_{current_round}.parquet"
)

# run models
m = RunModel(current_round=current_round)
m.run_foxhound()
m.run_deadcell()
m.run_cobra()
m.run_beautybeast()
m.run_skulls()
m.run_desperado()

# read model name json file
with open("data/model_names.json", "r") as f:
    model_names = json.load(f)

# submit predictions
for item in model_names.items():
    print(f"Submitting live predictions for {item[0]}...")
    napi.upload_predictions(
        file_path=f"predictions/{item[0]}_live_preds_{current_round}.csv",
        model_id=item[1]
    )

# complete
print(f"Submissions complete!")

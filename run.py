# run numerai models

# load libraries
import os
import gc
import json
import datetime
from numerapi import NumerAPI
from utils.inference import RunModel
from utils.models import GaiaModel

# instantiate env var
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')

# instantiate numerai client
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# check run
current_round = napi.get_current_round()
with open("data/run_log.json", "r") as f:
    last_run_log = json.load(f)

print(f"Current live round #   : {current_round}")
print(f"Last completed round # : {last_run_log['run_round']}")

# check run
if current_round == last_run_log["run_round"]:
    
    # end if no new round
    print("Already completed the current round!")

elif current_round > last_run_log["run_round"]:
    
    # begin process
    print("Begin submissions for new round.")

    # donwload live dataset
    napi.download_dataset(
        filename="v4/live.parquet", 
        dest_path=f"data/live_{current_round}.parquet"
    )
    
    # run models
    nmr = RunModel(roundn=current_round, mode="live")
    nmr.get_data()
    nmr.run_foxhound()
    nmr.run_deadcell()
    nmr.run_cobra()
    nmr.run_beautybeast()
    nmr.run_skulls()
    nmr.run_gaia()
    nmr.run_terra()
    nmr.run_spira()
    gc.collect()

    # run test models
    nmr.run_dojo()
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
    
    # log run
    run_log = {
        "run_date": datetime.datetime.now().strftime("%Y-%m-%d"), 
        "run_round": current_round
    }

    # save log
    with open("data/run_log.json", "w") as outfile:
        json.dump(run_log, outfile, indent = 4)

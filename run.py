# run numerai models

# load libraries
import os
import gc
import json
import datetime
from numerapi import NumerAPI
from utils.inference import RunModel
from utils.models import GaiaModel, SpiraModel

# instantiate env var
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')

# instantiate numerai client
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# check run
try:
    current_round = napi.get_current_round()
    print(f"Current live round #   : {current_round}")
except:
    pass

with open("predictions/run_log.json", "r") as f:
    last_run_log = json.load(f)
print(f"Last completed round # : {last_run_log['run_round']}")

# run if current round is live
try:
    if current_round == last_run_log["run_round"]:
        print("Already completed the current round!")

    elif current_round > last_run_log["run_round"]:
        print("Begin submissions for new round.")

        # donwload live dataset
        napi.download_dataset(
            filename="v4/live.parquet", 
            dest_path=f"data/live_{current_round}.parquet"
        )
        
        # instantiate
        nmr = RunModel(roundn=current_round, mode="live")

        # load data
        nmr.get_data()

        # run models
        nmr.run_foxhound()
        # nmr.run_deadcell()
        nmr.run_cobra()
        # nmr.run_beautybeast()
        # nmr.run_skulls()
        nmr.run_gaia()
        # nmr.run_terra()
        nmr.run_spira()
        gc.collect()

        # run test (dojo) model
        nmr.run_dojo()
        gc.collect()

        # read model name json file
        with open("data/model_names.json", "r") as f:
            model_names = json.load(f)

        # submit live predictions for current round
        for k, v in model_names.items():
            print(f"Submitting live predictions for {k}...")
            napi.upload_predictions(
                file_path=f"predictions/{k}_live_preds_{current_round}.csv",
                model_id=v
            )
        
        # log run
        print(f"Logging round submission record...")
        run_log = {
            "run_date": datetime.datetime.now().strftime("%Y-%m-%d"), 
            "run_round": current_round
        }

        # save log
        with open("predictions/run_log.json", "w") as outfile:
            json.dump(run_log, outfile, indent = 4)

        # delete prediction files
        print("Deleting submitted prediction files...")
        for k in model_names.keys():
            prediction_file_name = f"predictions/{k}_live_preds_{current_round}.csv"
            if os.path.exists(prediction_file_name):
                os.remove(prediction_file_name)
                print(f"Prediction file for {k} has been deleted!")
            else:
                print("Prediction file for {k} does not exist!")

        # complete
        print("Round complete!")

except:
    print("Something went wrong...")

# import dependencies
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import gc
import json
from utils.utils import (
    load_model,
    neutralize,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
)


# run model class
class RunModel:
    
    # initiate class
    def __init__(self, roundn, mode):

        self.roundn = roundn
        self.mode = mode

    # function to get data
    def get_data(self):
        
        if self.mode == "live":
            self.inference_data = pd.read_parquet(f'data/live_{self.roundn}.parquet').fillna(0.5)

        elif self.mode == "validation":
            self.inference_data = pd.read_parquet(f'data/validation.parquet')
    
    # function to save prediction
    def save_prediction(self, model_name, inference_data, model_to_submit):

        inference_data["prediction"] = inference_data[model_to_submit].rank(pct=True)

        if self.mode == "live":
            csv_name = f"predictions/{model_name}_live_preds_{self.roundn}.csv"

        elif self.mode == "validation":
             csv_name = f"predictions/{model_name}_val_preds.csv"

        inference_data["prediction"].to_csv(csv_name)

    # function to get feature names
    def get_features(self, get):
        if get == "all":
            with open("data/features.json", "r") as f: _features = json.load(f)
            return list(_features["feature_stats"].keys())

        elif get == "medium":
            with open("data/features.json", "r") as f: _features = json.load(f)
            return _features["feature_sets"]["medium"]

        elif get == "small":
            with open("data/features.json", "r") as f: _features = json.load(f)
            return _features["feature_sets"]["small"]

        elif get == "other":
            with open("data/features.json", "r") as f: _features = json.load(f)
            small = _features["feature_sets"]["small"]
            medium = _features["feature_sets"]["medium"]
            all = list(_features["feature_stats"].keys())
            return [x for x in all if x not in small and x not in medium]

        elif get == "fstats_500":
            with open("data/top_fstats_features.json", "r") as f: _features = json.load(f)
            return _features["top_500_features"]

        elif get == "top_bottom":
            with open("data/top_bottom_features.json", "r") as f: _features = json.load(f)
            return _features["top_features"] + _features["bottom_features"]

        elif get == "riskiest_50_medium":
            with open("data/riskiest_features.json", "r") as f: _features = json.load(f)
            return _features["riskiest_50_medium_features"]

        elif get == "riskiest_5_small":
            with open("data/riskiest_features.json", "r") as f: _features = json.load(f)
            return _features["riskiest_5_small_features"]

        elif get == "riskiest_60_other":
            with open("data/riskiest_features.json", "r") as f: _features = json.load(f)
            return _features["riskiest_60_other_features"]

        else:
            print("ERROR: Features list do not exist!")
    
    # function to run the foxhound model
    def run_foxhound(self):
        model_name = f"dh_foxhound"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_50_medium")
        model = load_model(model_name)
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = neutralize(
            df=inference_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_with_neutralization"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the deadcell model
    def run_deadcell(self):
        model_name = f"dh_deadcell"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="small")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_5_small")
        model = load_model(model_name)
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = neutralize(
            df=inference_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_with_neutralization"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the cobra model
    def run_cobra(self):
        model_name = f"dh_cobra"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="other")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_60_other")
        model = load_model(model_name)
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = neutralize(
            df=inference_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_with_neutralization"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")
    
    # function to run the beautybeast model
    def run_beautybeast(self):
        model_name = f"dh_beautybeast"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="fstats_500")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        model = load_model(model_name)
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        model_to_submit = f"preds_{model_name}"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the skulls model
    def run_skulls(self):
        model_name = f"dh_skulls"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="top_bottom")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        model = load_model(model_name)
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        model_to_submit = f"preds_{model_name}"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the desperado model
    def run_desperado(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.roundn}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_live_preds_{self.roundn}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.roundn}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_live_preds_{self.roundn}.csv")
        skulls_live = pd.read_csv(f"predictions/dh_skulls_live_preds_{self.roundn}.csv")
        features = ["foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        desperado_live = (
            foxhound_live
                .merge(right=deadcell_live, how='inner', on="id", suffixes=('', '2'))
                .merge(right=cobra_live, how='inner', on="id", suffixes=('', '3'))
                .merge(right=beautybeast_live, how='inner', on="id", suffixes=('', '4'))
                .merge(right=skulls_live, how='inner', on="id", suffixes=('', '5'))
        )
        desperado_live.columns = ["id", "foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.roundn}.csv")
        print(f"...model run complete!")
    
    # function to run the desperado model
    def run_desperadov3(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.roundn}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.roundn}.csv")
        features = ["foxhound", "cobra", "beautybeast"]
        desperado_live = (
            foxhound_live
                .merge(right=cobra_live, how='inner', on="id", suffixes=('', '2'))
                .merge(right=beautybeast_live, how='inner', on='id', suffixes=('', '3'))
        )
        desperado_live.columns = ["id"] + features
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.roundn}.csv")
        print(f"...model run complete!")
    
    # function to run the gaia model
    def run_gaia(self):
        model_name = f"dh_gaia"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_50_medium")
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = neutralize(
            df=inference_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_with_neutralization"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the terra model
    def run_terra(self):
        model_name = f"dh_terra"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_50_medium")
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = neutralize(
            df=inference_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_with_neutralization"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

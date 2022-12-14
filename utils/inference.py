# import dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import torch
import json
import scipy
from scipy.stats import skew

# load vars
ERA_COL = "era"
DATA_TYPE_COL = "data_type"
TARGET_COL = "target_nomi_v4_20"

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
    
    # function to neuralize preds (https://github.com/numerai/example-scripts/blob/master/utils.py)
    def run_neutralizer(self, df, columns, neutralizers=None, proportion=1.0, normalize=True, era_col=ERA_COL):
        if neutralizers is None:
            neutralizers = []
        unique_eras = df[era_col].unique()
        computed = []
        for u in unique_eras:
            df_era = df[df[era_col] == u]
            scores = df_era[columns].values
            if normalize:
                scores2 = []
                for x in scores.T:
                    x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
                    x = scipy.stats.norm.ppf(x)
                    scores2.append(x)
                scores = np.array(scores2).T
            exposures = df_era[neutralizers].values
            scores -= proportion * exposures.dot(
                np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(scores.astype(np.float32)))
            scores /= scores.std(ddof=0)
            computed.append(scores)
        return pd.DataFrame(np.concatenate(computed), columns=columns, index=df.index)

    # function to run the foxhound model
    def run_foxhound(self):
        model_name = f"dh_foxhound"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        riskiest_features = self.get_features(get="riskiest_50_medium")
        model = pd.read_pickle(f"models/{model_name}.pkl")
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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
        model = pd.read_pickle(f"models/{model_name}.pkl")
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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
        model = pd.read_pickle(f"models/{model_name}.pkl")
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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
        model = pd.read_pickle(f"models/{model_name}.pkl")
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
        model = pd.read_pickle(f"models/{model_name}.pkl")
        inference_data.loc[:, f"preds_{model_name}"] = model.predict(inference_data.loc[:, features])
        model_to_submit = f"preds_{model_name}"
        self.save_prediction(model_name, inference_data, model_to_submit)
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
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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

    # function to run the spira model
    def run_spira(self):
        model_name = f"dh_spira"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="fstats_500")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        infernece_ds = torch.from_numpy(inference_data[features].dropna().values)
        model = torch.load(f"models/{model_name}.pt")
        model.eval()
        inference_data.loc[:, f"preds_{model_name}"] = model(infernece_ds).squeeze(-1).detach().numpy()
        model_to_submit = f"preds_{model_name}"
        self.save_prediction(model_name, inference_data, model_to_submit)
        print(f"...model run complete!")

    # function to run the test model (currently testing Gaia v2)
    def run_dojo(self):
        model_name = f"dh_dojo"
        print(f"\nRunning {model_name} for live round # {self.roundn}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        inference_data = self.inference_data.loc[:, read_columns]
        infernece_ds = torch.from_numpy(inference_data[features].dropna().values)
        riskiest_features = self.get_features(get="riskiest_50_medium")
        model = torch.load(f"models/dh_gaia.pt") # hardcode
        model.eval()
        inference_data.loc[:, f"preds_{model_name}"] = model(infernece_ds).squeeze(-1).detach().numpy()
        inference_data[f"preds_{model_name}_with_neutralization"] = self.run_neutralizer(
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

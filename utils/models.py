# import dependencies
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import gc
import json
from utils.utils import (
    load_model,
    neutralize,
    get_biggest_change_features,
    ERA_COL,
    DATA_TYPE_COL,
    TARGET_COL,
)


# run model class
class RunModel:
    
    # initiate class
    def __init__(self, current_round):
        self.current_round = current_round

    # get data
    def get_data(self):
        self.training_data = pd.read_parquet('data/train.parquet')
        self.live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet').fillna(0.5)
    
    # get freature names
    def get_features(self, get):
        if get == "all":
            with open("data/features.json", "r") as f:
                _features = json.load(f)
            return list(_features["feature_stats"].keys())
        elif get == "medium":
            with open("data/features.json", "r") as f:
                _features = json.load(f)
            return _features["feature_sets"]["medium"]
        elif get == "small":
            with open("data/features.json", "r") as f:
                _features = json.load(f)
            return _features["feature_sets"]["small"]
        elif get == "other":
            with open("data/features.json", "r") as f:
                _features = json.load(f)
            small = _features["feature_sets"]["small"]
            medium = _features["feature_sets"]["medium"]
            all = list(_features["feature_stats"].keys())
            return [x for x in all if x not in small and x not in medium]
        elif get == "fstats_500":
            with open("data/top_fstats_features.json", "r") as f:
                _features = json.load(f)
            return _features["top_500_features"]
        elif get == "top_bottom":
            with open("data/top_bottom_features.json", "r") as f:
                _features = json.load(f)
            return _features["top_features"] + _features["bottom_features"]
        else:
            print("ERROR: Features list do not exist!")

    # function to run the foxhound model
    def run_foxhound(self, n_neutralize=50):
        model_name = f"dh_foxhound"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

    # function to run the deadcell model
    def run_deadcell(self, n_neutralize=5):
        model_name = f"dh_deadcell"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="small")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

    # function to run the cobra model
    def run_cobra(self, n_neutralize=60):
        model_name = f"dh_cobra"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="other")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")
    
    # function to run the beautybeast model
    def run_beautybeast(self):
        model_name = f"dh_beautybeast"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="fstats_500")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        live_data = self.live_data.loc[:, read_columns]
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        model_to_submit = f"preds_{model_name}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

    # function to run the skulls model
    def run_skulls(self):
        model_name = f"dh_skulls"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="top_bottom")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        live_data = self.live_data.loc[:, read_columns]
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        model_to_submit = f"preds_{model_name}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

    # function to run the desperado model
    def run_desperado(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.current_round}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.current_round}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_live_preds_{self.current_round}.csv")
        skulls_live = pd.read_csv(f"predictions/dh_skulls_live_preds_{self.current_round}.csv")
        features = ["foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        desperado_live = foxhound_live.merge(
            right=deadcell_live, how='inner', on="id", suffixes=('', '2')).merge(
            right=cobra_live, how='inner', on="id", suffixes=('', '3')).merge(
            right=beautybeast_live, how='inner', on="id", suffixes=('', '4')).merge(
            right=skulls_live, how='inner', on="id", suffixes=('', '5'))
        desperado_live.columns = ["id", "foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")
    
    # function to run the desperado model
    def run_desperadov3(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.current_round}.csv")
        features = ["foxhound", "cobra"]
        desperado_live = foxhound_live.merge(right=cobra_live, how='inner', on="id", suffixes=('', '2'))
        desperado_live.columns = ["id"] + features
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")
    
    # function to run the gaia model
    def run_gaia(self, n_neutralize=50):
        model_name = f"dh_gaia"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

    # function to run the terra model
    def run_terra(self, n_neutralize=50):
        model_name = f"dh_terra"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        features = self.get_features(get="medium")
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        print(f">>> Model {model_name} run complete!")

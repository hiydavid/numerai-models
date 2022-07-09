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
        self.live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet')
    
    # function to run the foxhound model
    def run_foxhound(self, n_neutralize=50):
        model_name = f"dh_foxhound"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["medium"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Neutralizing features ...")
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

    # function to run the deadcell model
    def run_deadcell(self, n_neutralize=5):
        model_name = f"dh_deadcell"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["small"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Neutralizing features ...")
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

    # function to run the cobra model
    def run_cobra(self, n_neutralize=60):
        model_name = f"dh_cobra"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        small_features = feature_metadata["feature_sets"]["small"]
        medium_features = feature_metadata["feature_sets"]["medium"]
        all_features = list(feature_metadata["feature_stats"].keys())
        features = [x for x in all_features if x not in small_features and x not in medium_features]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Neutralizing features ...")
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")
    
    # function to run the beautybeast model
    def run_beautybeast(self):
        model_name = f"dh_beautybeast"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/top_fstats_features.json", "r") as f:
            top_fstats_features = json.load(f)
        features = top_fstats_features["top_500_features"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

    # function to run the skulls model
    def run_skulls(self):
        model_name = f"dh_skulls"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/top_bottom_features.json", "r") as f:
            top_bottom_features = json.load(f)
        features = top_bottom_features["top_features"] + top_bottom_features["bottom_features"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

    # function to run the desperado model
    def run_desperado(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.current_round}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.current_round}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_live_preds_{self.current_round}.csv")
        skulls_live = pd.read_csv(f"predictions/dh_skulls_live_preds_{self.current_round}.csv")
        print(f">>> Preprocessing data ...")
        features = ["foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        desperado_live = foxhound_live.merge(
            right=deadcell_live, how='inner', on="id", suffixes=('', '2')).merge(
            right=cobra_live, how='inner', on="id", suffixes=('', '3')).merge(
            right=beautybeast_live, how='inner', on="id", suffixes=('', '4')).merge(
            right=skulls_live, how='inner', on="id", suffixes=('', '5'))
        desperado_live.columns = ["id", "foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        print(f">>> Creating live predictions ...")
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        gc.collect()
        print(f">>> Saving live predictions ...")
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")
    
    # function to run the desperadov2 model
    def run_desperadov2(self):
        """
        # In dev, not ready.
        """
        model_name = f"dh_desperadov2"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.current_round}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.current_round}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_live_preds_{self.current_round}.csv")
        skulls_live = pd.read_csv(f"predictions/dh_skulls_live_preds_{self.current_round}.csv")
        print(f">>> Preprocessing data ...")
        features = ["foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        live_data = foxhound_live.merge(
            right=deadcell_live, how='inner', on="id", suffixes=('', '2')).merge(
            right=cobra_live, how='inner', on="id", suffixes=('', '3')).merge(
            right=beautybeast_live, how='inner', on="id", suffixes=('', '4')).merge(
            right=skulls_live, how='inner', on="id", suffixes=('', '5'))
        live_data.columns = ["id"] + features
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = load_model(model_name)
        features = ["foxhound", "deadcell", "cobra", "beautybeast", "skulls"]
        live_data.loc[:, f"prediction"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Saving live predictions ...")
        live_data = live_data[["id", "prediction"]].set_index("id")
        live_data.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")
    
    # function to run the desperado model
    def run_desperadov3(self):
        model_name = f"dh_desperado"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_live_preds_{self.current_round}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_live_preds_{self.current_round}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_live_preds_{self.current_round}.csv")
        skulls_live = pd.read_csv(f"predictions/dh_skulls_live_preds_{self.current_round}.csv")
        print(f">>> Preprocessing data ...")
        features = ["foxhound", "cobra"]
        desperado_live = foxhound_live.merge(right=cobra_live, how='inner', on="id", suffixes=('', '2'))
        desperado_live.columns = ["id"] + features
        print(f">>> Creating live predictions ...")
        desperado_live["prediction"] = desperado_live[features].mean(axis=1)
        gc.collect()
        print(f">>> Saving live predictions ...")
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")
    
    # function to run the gaia model
    def run_gaia(self, n_neutralize=50):
        model_name = f"dh_gaia"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["medium"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Neutralizing features ...")
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

    # function to run the terra model
    def run_terra(self, n_neutralize=50):
        model_name = f"dh_terra"
        print(f"\nRunning {model_name} for live round # {self.current_round}...")
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["medium"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = self.training_data.loc[:, read_columns]
        live_data = self.live_data.loc[:, read_columns]
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(
            lambda era: era[features].corrwith(era[TARGET_COL])
        )
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        print(f">>> Loading model & creating live predictions ...")
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, features])
        gc.collect()
        print(f">>> Neutralizing features ...")
        live_data[f"preds_{model_name}_neutral_riskiest_{n_neutralize}"] = neutralize(
            df=live_data,
            columns=[f"preds_{model_name}"],
            neutralizers=riskiest_features,
            proportion=1.0,
            normalize=True,
            era_col=ERA_COL
        )
        print(f">>> Saving live predictions ...")
        model_to_submit = f"preds_{model_name}_neutral_riskiest_{n_neutralize}"
        live_data["prediction"] = live_data[model_to_submit].rank(pct=True)
        live_data["prediction"].to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()
        print(f">>> Model {model_name} run complete!")

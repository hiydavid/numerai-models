### Run Latest Foxhound Model

# import dependencies
import pandas as pd
from lightgbm import LGBMRegressor
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
    
    # function to run latest foxhound model
    def run_foxhound(self, n_neutralize=50, version=0):
        model_name = f"dh_foxhound_v{version}"
        print(f"\nBegin running {model_name} for live round # {self.current_round}...")

        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["medium"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = pd.read_parquet('data/train.parquet', columns=read_columns)
        live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet', columns=read_columns)
        
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda era: era[features].corrwith(era[TARGET_COL]))
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            total_rows = len(live_data[live_data["data_type"] == "live"])
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        
        print(f">>> Loading pre-trained model ...")
        model = load_model(model_name)
        model_expected_features = model.booster_.feature_name()
        
        print(f">>> Creating live predictions ...")
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, model_expected_features])
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

        print(f">>> Model {model_name} run complete for live round # {self.current_round}!")

    # function to run latest deadcell model
    def run_deadcell(self, n_neutralize=5, version=0):
        model_name = f"dh_deadcell_v{version}"
        print(f"\nBegin running {model_name} for live round # {self.current_round}...")
        
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["small"]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = pd.read_parquet('data/train.parquet', columns=read_columns)
        live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet', columns=read_columns)
        
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda era: era[features].corrwith(era[TARGET_COL]))
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            total_rows = len(live_data[live_data["data_type"] == "live"])
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        
        print(f">>> Loading pre-trained model ...")
        model = load_model(model_name)
        model_expected_features = model.booster_.feature_name()
        
        print(f">>> Creating live predictions ...")
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, model_expected_features])
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
        
        print(f">>> Model {model_name} run complete for live round # {self.current_round}!")

    # function to run latest cobra model
    def run_cobra(self, n_neutralize=60, version=0):
        model_name = f"dh_cobra_v{version}"
        print(f"\nBegin running {model_name} for live round # {self.current_round}...")
        
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        small_features = feature_metadata["feature_sets"]["small"]
        medium_features = feature_metadata["feature_sets"]["medium"]
        all_features = list(feature_metadata["feature_stats"].keys())
        features = [x for x in all_features if x not in small_features and x not in medium_features]
        read_columns = features + [ERA_COL, DATA_TYPE_COL, TARGET_COL]
        training_data = pd.read_parquet('data/train.parquet', columns=read_columns)
        live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet', columns=read_columns)
        
        print(f">>> Preprocessing data ...")
        all_feature_corrs = training_data.groupby(ERA_COL).apply(lambda era: era[features].corrwith(era[TARGET_COL]))
        riskiest_features = get_biggest_change_features(all_feature_corrs, n_neutralize)
        nans_per_col = live_data[live_data["data_type"] == "live"].isna().sum()
        if nans_per_col.any():
            total_rows = len(live_data[live_data["data_type"] == "live"])
            live_data.loc[:, features] = live_data.loc[:, features].fillna(0.5)
        else:
            pass
        gc.collect()
        
        print(f">>> Loading pre-trained model ...")
        model = load_model(model_name)
        model_expected_features = model.booster_.feature_name()
        
        print(f">>> Creating live predictions ...")
        live_data.loc[:, f"preds_{model_name}"] = model.predict(live_data.loc[:, model_expected_features])
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
        
        print(f">>> Model {model_name} run complete for live round # {self.current_round}!")

    # function to run latest beautybeast model
    def run_beautybeast(self, version=0):
        model_name = f"dh_beautybeast_v{version}"
        print(f"\nBegin running {model_name} for live round # {self.current_round}...")
        
        print(f">>> Importing data ...")
        with open("data/features.json", "r") as f:
            feature_metadata = json.load(f)
        features = feature_metadata["feature_sets"]["medium"]
        targets = [
            "target_nomi_v4_20", "target_jerome_v4_20", "target_janet_v4_20",
            "target_ben_v4_20", "target_alan_v4_20", "target_paul_v4_20",
            "target_george_v4_20", "target_william_v4_20", "target_arthur_v4_20",
            "target_thomas_v4_20"
        ]
        read_columns = features + targets + [ERA_COL, DATA_TYPE_COL]
        training_data = pd.read_parquet('data/train.parquet', columns=read_columns)
        live_data = pd.read_parquet(f'data/live_{self.current_round}.parquet', columns=read_columns)
        
        print(f">>> Preprocessing data ...")
        aux_targets = [col for col in training_data.columns if col.endswith("_20") and col != "target_nomi_v4_20"]
        main_target = "target_nomi_v4_20"
        gc.collect()
        
        print(f">>> Loading pre-trained model ...")
        model_list = []
        for t in aux_targets:
            model = load_model(t)
        model_list.append(model)

        print(f">>> Creating live predictions ...")
        live_preds_list = []
        for t, m in zip(aux_targets, model_list):
            live_preds = pd.DataFrame(m.predict(live_data[features])).rename(columns={0:f"{t}"})
            live_preds_list.append(live_preds)
        live_preds_all = pd.concat(live_preds_list, axis=1)
        live_preds_avg_ranked = live_preds_all.mean(axis=1).rank(pct=True, method="first")
        gc.collect()

        print(f">>> Saving live predictions ...")
        live_preds = pd.DataFrame(live_preds_avg_ranked).rename(columns={0:"prediction"}).set_index(live_data.index)
        live_preds.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()

        print(f">>> Model {model_name} run complete for live round # {self.current_round}!")

    # function to run latest beautybeast model
    def run_desperado(self, version=0):
        model_name = f"dh_desperado_v{version}"
        print(f"\nBegin running {model_name} for live round # {self.current_round}...")
        
        print(f">>> Importing data ...")
        foxhound_live = pd.read_csv(f"predictions/dh_foxhound_v0_live_preds_{self.current_round}.csv")
        deadcell_live = pd.read_csv(f"predictions/dh_deadcell_v0_live_preds_{self.current_round}.csv")
        cobra_live = pd.read_csv(f"predictions/dh_cobra_v0_live_preds_{self.current_round}.csv")
        beautybeast_live = pd.read_csv(f"predictions/dh_beautybeast_v0_live_preds_{self.current_round}.csv")
        
        print(f">>> Preprocessing data ...")
        desperado_live = foxhound_live.merge(
            right=deadcell_live, how='inner', on="id", suffixes=('', '2')).merge(
            right=cobra_live, how='inner', on="id", suffixes=('', '3')).merge(
            right=beautybeast_live, how='inner', on="id", suffixes=('', '4'))
        desperado_live.columns = ["id", "foxhound", "deadcell", "cobra", "beautybeast"]

        print(f">>> Creating live predictions ...")
        desperado_live["prediction"] = (
            desperado_live["foxhound"] + 
            desperado_live["deadcell"] + 
            desperado_live["cobra"] +
            desperado_live["beautybeast"]
            ) / 4
        gc.collect()
        
        print(f">>> Saving live predictions ...")
        desperado_live = desperado_live[["id", "prediction"]].set_index("id")
        desperado_live.to_csv(f"predictions/{model_name}_live_preds_{self.current_round}.csv")
        gc.collect()

        print(f">>> Model {model_name} run complete for live round # {self.current_round}!")

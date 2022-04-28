# SCRIPT FOR MODEL EVALUATION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerapi import NumerAPI
from api_keys import PUBLIC_ID, SECRET_KEY

# instantiate
napi = NumerAPI(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

# create numerai_score function
def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# regular pearson corr
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]

# get model performance history
def get_model_rankings(model_list):
    df_list = []
    for m in model_list:
        df = pd.DataFrame.from_dict(napi.daily_model_performances(m))
        df['modelName'] = m
        df_list.append( df)
    df_all = pd.concat(df_list).set_index("date")
    return df_all

# plot model rankings
def plot_model_rankings(data, after_date, select_rank):
    data[after_date:].groupby('modelName')[select_rank].plot(
        y=select_rank,
        title=select_rank,
        figsize=(10, 3),
        legend=True
    )
    plt.show()
    
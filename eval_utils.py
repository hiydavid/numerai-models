# SCRIPT FOR MODEL EVALUATION

# create numerai_score function
def numerai_score(y_true, y_pred):
    rank_pred = y_pred.groupby(eras).apply(lambda x: x.rank(pct=True, method="first"))
    return np.corrcoef(y_true, rank_pred)[0,1]

# regular pearson corr
def correlation_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0,1]
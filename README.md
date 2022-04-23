# Models for Numerai Competition

## Models
### LightGBM Regressor
* Foxhound_v0:
  * Trained on all medium features
  * Top 50 riskiest features neuralized
  * Params: `{"n_estimators": 2000, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 2 ** 5, "colsample_bytree": 0.1}`

* Deadcell_v0:
  * Trained on all small features
  * Top 5 riskiest features neutralized
  * Params: `{"n_estimators": 1000, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 2 ** 5, "colsample_bytree": 0.1`

* CobraUnit_v0:
  * d

* BeautyBeast_v0:
  * d

* Skulls_v0:
  * d

### Deep Neural Networks
* *incoming*

### Voting Regressor
* *incoming*

## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)


## Helpful Links
* [Tournament documentation](https://docs.numer.ai/)
* [Numerai example script](https://github.com/numerai/example-scripts)

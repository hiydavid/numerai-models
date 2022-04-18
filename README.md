# Models for Numerai Competition

## Models
### LightGBM Regressor
* base_v0 :
  * Trained on top 50 riskiest "medium" features
  * Params:
    * "n_estimators": 2000
    * "learning_rate": 0.01
    * "max_depth": 5
    * "num_leaves": 2 ** 5
    * "colsample_bytree": 0.1

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

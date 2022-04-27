# Models for Numerai Competition

## Models
### **Foxhound**: Medium features, with 50 neuralized
* v0. LightGBM params:
  * `"n_estimators": 2000`
  * `"learning_rate": 0.01`
  * `"max_depth": 5`
  * `"num_leaves": 2 ** 5`
  * `"colsample_bytree": 0.1`

### **Deadcell**: Small features, with 5 neutralized
* v0. LightGBM params:
  * `"n_estimators": 1000`
  * `"learning_rate": 0.01`
  * `"max_depth": 5`
  * `"num_leaves": 2 ** 5`
  * `"colsample_bytree": 0.1`
* v1. XGBoost params:
  * TBD

### **Cobra**: All features, with 120 neutralized
* v0. TBD, params:

### **BeautyBeast**:

### **Skulls**:

### **Desperado**: Ensemble prediction of all other models
* v0. Simple averaging

## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)

## Helpful Links
* [Tournament documentation](https://docs.numer.ai/)
* [Numerai example script](https://github.com/numerai/example-scripts)

## Project Roadmap (By June 30th)
* Finish research issues
* Complete building out new models
  1. Rebuild *Deadcell* with XGBoost
  2. Build out *Cobra*, an all features model
  3. Build out *BeautyBeast*, ensemble model trained on alternative features
  4. Build out *Skulls*, an neural network based model trained on all features
  5. Build a new version of *Desperado* with stacking regressor
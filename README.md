# Models for Numerai Competition

## Performance
<img width="550" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/ModelCorrPlots.png">
<img width="550" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/ModelTCPlots.png">
<img width="550" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/SharpeRatioPlots.png">
<img width="550" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/TotalReturnPlots.png">

## Models
### **LightGBM-Based Models**
* **Foxhound**: Trained on medium features, with top 50 features neuralized

* **Deadcell**: Trained on small features, with top 5 features neutralized

* **Cobra**: Trained on rest of the features, with top 60 features neutralized

* **BeautyBeast**: Trained on top 500 features by f-stats, without neutralization

* **Skulls**: Trained on top 200 and bottom 200 features by target correlation, without neutralization

* [Decommissioned] **Desperado**: Ensemble prediction using simple averaging across Foxhound, Cobra, and BeautyBeast

### **DNN-Based Models**
* **Gaia**: Multilayer Perceptron network with 4 layers, with 472, 235, 118, 59 units each respectively, trained on medium features, with top 50 features neuralized

* **Terra**: Deep-Cross network with 2 layers, with 472, 236 units each respectively, feature crossing layer with batch normalization, trained on medium features, with top 50 features neuralized

* **Spira**: WIP

* **Ivalice**: WIP

* **Cocoon**: WIP

* **Eos**: WIP

## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)

## Project Roadmap (By 2022/12/31)
* Create and implement a model using numeric embedding of features
* Create and implement a model using attention-based architecture

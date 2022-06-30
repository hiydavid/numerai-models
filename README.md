# Models for Numerai Competition

## Models
### **LightGBM-Based Models**

<img width="706" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/CorrRankPlots.png">
<img width="706" alt="image" src="https://raw.githubusercontent.com/hiydavid/numerai-models/main/plots/PayoutPlots.png">

#### Foxhound: 
* Trained on medium features and main target
* 50 features neuralized

#### Deadcell: 
* Trained on small features and main target
* 5 features neutralized

#### Cobra: 
* Trained on rest of the features and main target
* 60 features neutralized

#### BeautyBeast: 
* Trained on medium features and auxiliary targets
* Ensemble output as simple avgeraging

#### Skulls:
* Trained on top 200 and bottom 200 features
* Top/bottom determined by correlation to target
* No features neutralized

#### Desperado: 
* Ensemble prediction of all other models
* v1 using simple averaging
* v2 using LightGBM, trained on 50% of combined validation set

### DNN-Based Models

#### Gaia
* MLP with 4 layers, with 472, 235, 118, 59 units each respectively
* Trained on medium features and main target
* 50 features neuralized

#### Terra
* MLP with 2 layers, with 472, 236 units each respectively
* Added a feature crossing layer with batch normalization
* Trained on medium features and main target
* 50 features neuralized

#### Spira
#### Ivalice
#### Cocoon
#### Eos


## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)

## Project Roadmap (By 2022/12/31)
* Create and implement a model using DCN architecture
* Create and implement a model using numeric embedding of features
* Create and implement a model using attention-based architecture

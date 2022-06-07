# Models for Numerai Competition

## Models
### **LightGBM-Based Models**

<img width="706" alt="image" src="https://user-images.githubusercontent.com/42119351/172471486-8150c31e-1280-40fc-9629-37d448c33ad5.png">
<img width="706" alt="image" src="https://user-images.githubusercontent.com/42119351/172475526-0b9b1ad9-42c9-481b-a3df-90d981f7ab5b.png">

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
* Simple averaging

### DNN-Based Models

#### Gaia
#### Terra 
#### Spira
#### Ivalice
#### Cocoon
#### Eos


## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)

## Helpful Links
* [Tournament documentation](https://docs.numer.ai/)
* [Numerai example script](https://github.com/numerai/example-scripts)

## Project Roadmap (By June 30th)
* Finish research issues
* Complete building out new models
  1. Build a new version of *Desperado* with stacking regressor

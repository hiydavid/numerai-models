# Models for Numerai Competition

## Models
### **Foxhound**: 
* Trained on medium features and main target
* 50 neuralized
* LightGBM model

### **Deadcell**: 
* Trained on small features and main target
* 5 neutralized
* LightGBM model

### **Cobra**: 
* Trained on rest of the features and main target
* 60 neutralized
* LightGBM model

### **BeautyBeast**: 
* Trained on medium features and auxiliary targets
* Ensemble output as simple avgeraging
* Auxiliary models using LightGBM

### **Skulls**:
* Trained on top 200 and bottom 200 features
* Top/bottom determined by correlation to target
* No features neutralized
* LightGBM model

### **Desperado**: 
* Ensemble prediction of all other models
* Simple averaging

---
## Tracking Results
* [Model performance](https://numer.ai/models)
* [Tournament leaderboard](https://numer.ai/tournament)

---
## Helpful Links
* [Tournament documentation](https://docs.numer.ai/)
* [Numerai example script](https://github.com/numerai/example-scripts)

---
## Project Roadmap (By June 30th)
* Finish research issues
* Complete building out new models
  1. Build out *Skulls*, an neural network based model trained on all features
  2. Build a new version of *Desperado* with stacking regressor
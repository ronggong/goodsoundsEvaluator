# Good-sounds Evaluator
Deep learning models for mono musical note assessment

## Usage
To run the pre-trained Keras models, please refer to `demo_CNNs_models.ipynb`

|                   | Accuracy     |
|-------------------|--------------|
| Timbre stability  | 0.9249097473 |
| Pitch stability   | 0.9415162455 |
| Dynamic stability | 0.9682310469 |
| Timbre richness   | 0.9321299639 |
| attack clarity    | 0.9602888087 |

To run the pre-trained XGBoost models, please refer to `demo_xgboost_models.ipynb`

|                   | Accuracy     |
|-------------------|--------------|
| Timbre stability  | 0.9386281588 |
| Pitch stability   | 0.9364620939 |
| Dynamic stability | 0.9227436823 |
| Timbre richness   | 0.9487364621 |
| attack clarity    | 0.9422382671 |

The running speed of each models is indicated in each `ipynb`.

## Requirements
Keras>1.2.1  
Theano>0.8.2  
h5py>2.7.0  
pandas>0.17.0
xgboost==0.6
numpy  
Essentia 2.1-beta3  
scikit-learn==0.18.1

[Instruction on compiling XGBoost](https://github.com/dmlc/xgboost/blob/master/doc/build.md)

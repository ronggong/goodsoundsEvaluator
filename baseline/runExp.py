from filePath import *
from subprocess import call


# # train model put in classifier folder
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_timbre_train.csv','train','timbre'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_pitch_train.csv','train','pitch'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_dynamics_train.csv','train','dynamics'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_richness_train.csv','train','richness'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_attack_train.csv','train','attack'])

# test

# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_timbre_test.csv','test','timbre'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_pitch_test.csv','test','pitch'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_dynamics_test.csv','test','dynamics'])
# call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_richness_test.csv','test','richness'])
call(['python','xgb_classification.py','../dataset/bag-of-feature/feature_attack_test.csv','test','attack'])

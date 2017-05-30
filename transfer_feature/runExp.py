from filePath import *
from subprocess import call


# # train model put in classifier folder
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_timbre.csv','train','timbre_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_pitch.csv','train','pitch_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_dynamics.csv','train','dynamics_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_richness.csv','train','richness_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_attack.csv','train','attack_gru_1layer'])

# test

# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/testData_timbre.csv','test','timbre_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/testData_pitch.csv','test','pitch_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/testData_dynamics.csv','test','dynamics_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/testData_richness.csv','test','richness_gru_1layer'])
# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/testData_attack.csv','test','attack_gru_1layer'])

# call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_pitch_nogru.csv','train','pitch_nogru'])
call(['python','../baseline/xgb_classification.py','../dataset/transfer_feature/trainData_pitch_nogru.csv','test','pitch_nogru'])

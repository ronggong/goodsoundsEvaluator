from os.path import join

path_dataset        = '/Users/gong/Documents/MTG document/dataset/goodsound'
path_sound_files    = join(path_dataset, 'sound_files')
path_csv_files      = join(path_dataset, 'csv')

path_csv_good_sounds_file       = join(path_csv_files, 'good_sounds.csv')
path_csv_bad_timbre_file       = join(path_csv_files, 'bad_timbre.csv')
path_csv_bad_richness_file       = join(path_csv_files, 'bad_richness.csv')
path_csv_bad_pitch_file       = join(path_csv_files, 'bad_pitch.csv')
path_csv_bad_dynamics_file       = join(path_csv_files, 'bad_dynamics.csv')
path_csv_bad_attack_file       = join(path_csv_files, 'bad_attack.csv')


import os
dir_path = os.path.dirname(os.path.realpath(__file__))
scaler_path = join(dir_path,'./baseline/models')
classifier_path = join(dir_path,'./baseline/models')
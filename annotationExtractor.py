import csv

def filepathKlassExtraction(csv_annotation_file):
    filenames = []
    with open(csv_annotation_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        csv_headings    = next(reader)
        # print(csv_headings)
        idx_filename    = csv_headings.index('filename')

        for row in reader:
            filenames.append(row[idx_filename])

    return filenames

if __name__ == '__main__':
    from filePath import *
    filenames_good_sounds = filepathKlassExtraction(path_csv_good_sounds_file)
    filenames_bad_timbre = filepathKlassExtraction(path_csv_bad_timbre_file)
    filenames_bad_richness = filepathKlassExtraction(path_csv_bad_richness_file)
    filenames_bad_attack = filepathKlassExtraction(path_csv_bad_attack_file)
    filenames_bad_pitch = filepathKlassExtraction(path_csv_bad_pitch_file)
    filenames_bad_dynamics = filepathKlassExtraction(path_csv_bad_dynamics_file)

    filenames_all = filenames_good_sounds+\
                    filenames_bad_attack+\
                    filenames_bad_dynamics+\
                    filenames_bad_pitch+\
                    filenames_bad_richness+\
                    filenames_bad_timbre

    filenames_all = list(set(filenames_all))

    # [timbre, pitch, dynamics, richness, attack]
    dict_labeling = {}
    for fn in filenames_all:
        dict_labeling[fn] = [1,1,1,1,1]

    for fn in filenames_bad_timbre:
        dict_labeling[fn][0] = 0

    for fn in filenames_bad_pitch:
        dict_labeling[fn][1] = 0

    for fn in filenames_bad_dynamics:
        dict_labeling[fn][2] = 0

    for fn in filenames_bad_richness:
        dict_labeling[fn][3] = 0

    for fn in filenames_bad_attack:
        dict_labeling[fn][4] = 0

    import json
    json.dump(dict_labeling, open('test_rong.json', 'wb'))
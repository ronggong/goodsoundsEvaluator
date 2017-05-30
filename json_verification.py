import json

with open('test_oriol_4.json') as data_file:
    label_oriol = json.load(data_file)

with open('test_rong.json') as data_file:
    label_rong = json.load(data_file)

print(len(label_oriol))
print(len(label_rong))

for key in label_rong:
    if label_oriol[key][0] != label_rong[key]:
        print(key)
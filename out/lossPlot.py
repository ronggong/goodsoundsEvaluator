import csv
import os
import matplotlib.pyplot as plt

path_output = './bag-of-feature-framelevel'

filename_output_1 = os.path.join(path_output,'rnn_model_timbre_gru1layer_32nodes_variable_length.csv')
filename_output_2 = os.path.join(path_output,'rnn_model_timbre_gru1layer_bidirectional_128nodes_variable_length.csv')

def outputCsvReader(filename_csv):
    """
    read loss output into a list
    :param filename_csv:
    :return:
    """
    data = []
    with open(filename_csv, "rb") as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            data.append(row)
    return data


def extractLoss(data):
    """
    organize the output into a dictionary
    :param data:
    :return:
    """
    data_loss = {'epoch':[], 'train_loss':[], 'val_loss':[]}
    for line in data:
        for ii in range(len(line)):
            if line[ii].strip() == 'epoch':
                data_loss['epoch'].append(int(line[ii+1]))
            elif line[ii].strip() == 'train_loss':
                data_loss['train_loss'].append(float(line[ii+1]))
            elif line[ii].strip() == 'val_loss':
                data_loss['val_loss'].append((float(line[ii+1])))
    return data_loss

def trainValLossPlot(dict_data_loss):
    plt.figure()
    plt.plot(dict_data_loss['epoch'], dict_data_loss['train_loss'], label='train_loss')
    plt.plot(dict_data_loss['epoch'], dict_data_loss['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('train loss and validation loss')
    plt.show()

def twoValLossPlot(dict_data_loss_1, dict_data_loss_2):
    epoch_1 = dict_data_loss_1['epoch']
    epoch_2 = dict_data_loss_2['epoch']

    len_min = min(len(epoch_1), len(epoch_2))

    plt.figure()
    plt.plot(dict_data_loss_1['epoch'][:len_min], dict_data_loss_1['val_loss'][:len_min], label='val_loss_1')
    plt.plot(dict_data_loss_2['epoch'][:len_min], dict_data_loss_2['val_loss'][:len_min], label='val_loss_2')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('val loss')
    plt.show()

data_1 = outputCsvReader(filename_output_1)
dict_data_loss_1 = extractLoss(data_1)

data_2 = outputCsvReader(filename_output_2)
dict_data_loss_2 = extractLoss(data_2)

trainValLossPlot(dict_data_loss_1)
trainValLossPlot(dict_data_loss_2)
twoValLossPlot(dict_data_loss_1, dict_data_loss_2)

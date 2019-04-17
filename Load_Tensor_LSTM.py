import torch
import glob


print('Loading data')
data_train = []
target_train = []
data_valid = []
target_valid = []
data_test = []
target_test = []
compteur = 0
for im_path in glob.glob("C:/Users/DenisCorbin/Desktop/LSTM_Input/LSTM_Target*.pt"):
    dataStr = im_path[im_path.find('Target_') +7 :-1] + 't'
    im_pathData = 'C:/Users/DenisCorbin/Desktop/LSTM_Input/LSTM_Data_' + dataStr
    if compteur < 100 :
        target_train.append(torch.load(im_path))
        data_train.append(torch.load(im_pathData))
    if compteur >= 100 and compteur < 122:
        target_valid.append(torch.load(im_path))
        data_valid.append(torch.load(im_pathData))
    if compteur >= 122:
        target_test.append(torch.load(im_path))
        data_test.append(torch.load(im_pathData))
    compteur += 1
print('Done loading data')
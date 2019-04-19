# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:23:12 2019

@author: talmezh
"""
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import imageio
import glob

comb_fact = 0.2
itteration_data = 1
itteration_target = 1
save = 0

class CNNEncoderDecoderMoreFeatures(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoderMoreFeatures, self).__init__()
        self.itteration_data = itteration_data
        self.itteration_target = itteration_target
        self.type = ''
        self.name = "Encoder Decoder CNN with more features"
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(2).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(2).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(4).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(4).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(8).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(8).double(),
            nn.ReLU(inplace=True)

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(8).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(4).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(4).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(2).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(2).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True)

        )
    def forward(self, x):
        x = self.encoder(x)
        if save:
            if typeTarget:
                torch.save(x, 'C:/Users/Denis/Desktop/LSTM_Input/LSTM_' +'Target_' + str(self.itteration_target).zfill(3) +'.pt')
                self.itteration_target += 1
            if typeData:
                torch.save(x, 'C:/Users/Denis/Desktop/LSTM_Input/LSTM_' +'Data_' + str(self.itteration_data).zfill(3) +'.pt')
                self.itteration_data += 1
        x = self.decoder(x)
        return x/x.max()

# %%
class LSTM_predictor(nn.Module):
    def __init__(self):
        super(LSTM_predictor, self).__init__()
        self.name = "LSTM"
#        self.h0 = torch.randn(3,1,300) #Bonne init?
#        self.c0 = torch.randn(3,1,300)
        self.lstm = nn.Sequential(
                nn.LSTM(300,300,3)
        )
    def forward(self, image_seq):
        output, (hn,cn) = self.lstm(image_seq)
        return output[-1] #Possiblement besoin de reshape en 15,20 selon le target
    
# %%
def LLE(output, maxY_t,max_X_t):    
    pixel_sum = output.sum()
    
    #Calcul du x predit
    v_x = torch.arange(output.size(3)).double() #Vecteur allant de 0 a 639
    partial_sumx = torch.mv(output.squeeze(),v_x).sum() #Somme ponderee selon x
#    for x in range(output.size(3)):
#        partial_sumx_long += (output[0, 0, :, x].sum()) * x
    x_pred = partial_sumx / pixel_sum
    
    output_T = torch.transpose(output,2,3) #Transposition de la matrice
    v_y = torch.arange(output.size(2)).double()
    partial_sumy = torch.mv(output_T.squeeze(),v_y).sum()
#    partial_sumy_long = 0
#    for y in range(output.size(2)):
#        partial_sumy_long += (output[0, 0, y, :].sum()) * y
    y_pred = partial_sumy / pixel_sum

    LLE = torch.sqrt((torch.from_numpy(max_X_t).double() - x_pred).pow(2) + (torch.from_numpy(maxY_t).double() - y_pred).pow(2))
#    print(LLE.item(), max_X_t, maxY_t, x_pred.item(), y_pred.item())
    return LLE
# %%

def test(CNN, LSTM, test_data, test_data_cnn, test_target):
    CNN.eval()
    LSTM.eval()
    test_loss = 0
    correct = 0
    dataSize = len(test_data)
    for index in range(len(test_data)):
        previous_f = Variable(test_data[index]) #tensor(3,1,300)
        current_frame = Variable(test_data_cnn[index]) #tensor(1,1,480,640)
        target = Variable(test_target[index])
        
        ft0_CNN = CNN.encoder(current_frame).view(8,300) #tensor(300)
        
        ft0_LSTM = LSTM(CNN.encoder(previous_f).view(3,8,300).float()).double() #tensor(300)
        
        ft0 = comb_fact*ft0_LSTM + (1-comb_fact)*ft0_CNN
        ft0 = ft0.view(1,8,15,20)
        
        output = CNN.decoder(ft0)
        output = output/output.max()
        
        maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target) + LLE(output, maxY_t,max_X_t)
        test_loss += loss.item()  # sum up batch loss
        if loss.item() <= 19:
            correct += 1;
    test_loss /= dataSize
    print('\n' + "Test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataSize, 100. * correct / dataSize))

def testCNN(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    dataSize = len(test_loader)
    for batch_idx in range(dataSize):
          #        data, target = Variable(valid_loader[batch_idx], volatile=True).cuda(),Variable(target_valid[batch_idx]).cuda() # if you have access to a gpu
        data, target = Variable(test_loader[batch_idx], volatile=True), Variable(target_test_CNN[batch_idx])
        output = model(data)
        maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target) + LLE(output, maxY_t,max_X_t)
        test_loss += loss.item()  # sum up batch loss
        if loss.item() <= 15:
            correct += 1;
    test_loss /= dataSize
    print('\n' + "Test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataSize,
        100. * correct / dataSize))
# %%
#Load models
CNN = CNNEncoderDecoderMoreFeatures()
CNN.load_state_dict(torch.load('best_ED_300_mb1_morefeatures.pth'))

LSTM = LSTM_predictor()
LSTM.load_state_dict(torch.load('best_model_LSTM_batch1_morefeatures_test2.pth'))

#Load data
print('Loading data')
data_test = []
target_test = []
data_test_CNN = []
target_test_CNN = []
compteur = 0 
fileNameTarget  = glob.glob("C:/Users/Denis/Desktop/CIL1/Annotation/Output0/*.npy")
fileNameData= []
for im_path in fileNameTarget:
    dataStr = im_path[im_path.find('\\') + 1:im_path.find('\\') + 5]
    indice = int(dataStr)
    im_pathData = 'C:/Users/Denis/Desktop/CIL1/Data/' + dataStr + '.png'
    if compteur >= 122:
        filePathData2 = 'C:/Users/Denis/Desktop/CIL1/Data/' + str(indice-1).zfill(4) + '.png'
        filePathData3 = 'C:/Users/Denis/Desktop/CIL1/Data/' + str(indice-2).zfill(4) + '.png'
        filePathData4 = 'C:/Users/Denis/Desktop/CIL1/Data/' + str(indice-3).zfill(4) + '.png'
        x = torch.from_numpy(np.array(imageio.imread(filePathData4)) / 255).view(1, 1, 480, 640).double()
        y = torch.from_numpy(np.array(imageio.imread(filePathData3)) / 255).view(1, 1, 480, 640).double()
        z = torch.from_numpy(np.array(imageio.imread(filePathData2)) / 255).view(1, 1, 480, 640).double()
        data_test.append(torch.cat([x,y,z]))
        target_test_CNN.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_test_CNN.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    compteur += 1
print('Done loading data')

#Run test
testCNN(CNN,data_test_CNN)
test(CNN,LSTM,data_test,data_test_CNN,target_test_CNN)
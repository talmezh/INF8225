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
import Add_gaussian.py


#%%
def label2MaskMap(data, c_dx = 0, c_dy = 0, shape=(512,512), radius = 10, normalize = False):
    """
    Generate a Mask map from the coordenates
    :param M, N: dimesion of output
    :param position: position of the label
    :param radius: is the radius of the gaussian function
    :return: a MxN normalized array
    """

    # Our 2-dimensional distribution will be over variables X and Y
    (M,N) = (shape[0], shape[1])
    #if len(data)<=2:
    #    data = [data]

    maskMap = []
    for index, value in enumerate(data):
        x,y = value

        #Correct the labels
        x = x + c_dx
        y = y + c_dy

        X = np.linspace(0, M - 1, M)
        Y = np.linspace(0, N - 1, N)
        X, Y = np.meshgrid(X, Y)
        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        # Mean vector and covariance matrix
        mu = np.array([x, y])
        Sigma = np.array([[radius, 0], [0, radius]])

        # The distribution on the variables X, Y packed into pos.
        Z = multivariate_gaussian(pos, mu, Sigma)

        # Normalization
        if normalize:
            Z = Z * (1 / np.max(Z))
        else:
            # 8bit image values (the loss go to inf+)
            Z = Z * (1 / np.max(Z))
            Z = np.asarray(Z * 255, dtype=np.uint8)

        maskMap.append(Z)

    if len(maskMap) == 1:
        maskMap = maskMap[0]

    return np.asarray(maskMap)

def multivariate_gaussian(pos, mu, Sigma):
    """
    Return the multivariate Gaussian distribution on array.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

#%%

comb_fact = 0
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
                torch.save(x, 'C:/Users/DenisCorbin/Desktop/LSTM_Input/LSTM_' +'Target_' + str(self.itteration_target).zfill(3) +'.pt')
                self.itteration_target += 1
            if typeData:
                torch.save(x, 'C:/Users/DenisCorbin/Desktop/LSTM_Input/LSTM_' +'Data_' + str(self.itteration_data).zfill(3) +'.pt')
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

pos_centre =[]

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
    pos_centre.append([x_pred.item(), y_pred.item()])
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
        if loss.item() <= 15:
            correct += 1;
    test_loss /= dataSize
    print('\n' + "Test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataSize, 100. * correct / dataSize))
    return test_loss, 100. * correct / dataSize

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
    return test_loss, 100. * correct / dataSize
# %%
#Load models
CNN = CNNEncoderDecoderMoreFeatures()
CNN.load_state_dict(torch.load('best_ED_300_mb1_morefeatures.pth'))

LSTM = LSTM_predictor()
LSTM.load_state_dict(torch.load('best_model_LSTM_batch20_morefeatures.pth'))

#Load data
print('Loading data')
data_test = []
target_test = []
data_test_CNN = []
target_test_CNN = []
compteur = 0 
fileNameTarget  = glob.glob("C:/Users/DenisCorbin/Desktop/CIL1/Annotation/Output0/*.npy")
fileNameData= []
for im_path in fileNameTarget:
    dataStr = im_path[im_path.find('\\') + 1:im_path.find('\\') + 5]
    indice = int(dataStr)
    im_pathData = 'C:/Users/DenisCorbin/Desktop/CIL1/Data/' + dataStr + '.png'
    if compteur >= 122:
        filePathData2 = 'C:/Users/DenisCorbin/Desktop/CIL1/Data/' + str(indice-1).zfill(4) + '.png'
        filePathData3 = 'C:/Users/DenisCorbin/Desktop/CIL1/Data/' + str(indice-2).zfill(4) + '.png'
        filePathData4 = 'C:/Users/DenisCorbin/Desktop/CIL1/Data/' + str(indice-3).zfill(4) + '.png'
        x = torch.from_numpy(np.array(imageio.imread(filePathData4)) / 255).view(1, 1, 480, 640).double()
        y = torch.from_numpy(np.array(imageio.imread(filePathData3)) / 255).view(1, 1, 480, 640).double()
        z = torch.from_numpy(np.array(imageio.imread(filePathData2)) / 255).view(1, 1, 480, 640).double()
        data_test.append(torch.cat([x,y,z]))
        target_test_CNN.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_test_CNN.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    compteur += 1
print('Done loading data')

#%% Run test
#testCNN(CNN,data_test_CNN)
x = []
avg_loss_tab = []
accuracy_tab = []
comb_fact = 0
for i in range (1000):
    print(comb_fact)
    x.append(comb_fact)
    avg_loss, accuracy = test(CNN,LSTM,data_test,data_test_CNN,target_test_CNN)
    avg_loss_tab.append(avg_loss)
    accuracy_tab.append(accuracy)
    comb_fact += 0.01

plt.figure(1)
ax1 = plt.subplot(111)
ax1.plot(x,avg_loss_tab)
ax1.set_title('Pertes selon le facteur de combinaison entre le LSTM et le CNN')
ax1.set_ylabel('LLE & MSE')
ax1.set_xlabel('Facteur de combinaison')
ax1.legend(['test'])
plt.figure(2)
ax2 = plt.subplot(111)
ax2.plot(x,accuracy_tab, 'k')
ax2.set_title('Précision selon le facteur de combinaison entre le LSTM et le CNN')
ax2.set_ylabel('%')
ax2.set_xlabel('Facteur de combinaison')
ax2.legend(['précision'])
plt.show()

#%%
test(CNN,LSTM,data_test,data_test_CNN,target_test_CNN)
out1 = CNN.encoder(data_test[0])
out2 = LSTM(out1.view(3,8,300).float())
out3 = CNN.decoder(out2.view(1,8,15,20).double())
image = out3.detach().numpy().squeeze()
target_test = torch.from_numpy(np.load("C:/Users/DenisCorbin/Desktop/CIL1/Annotation/Output0/0001.npy")).view(1, 1, 480, 640).double()
image2 = target_test[0].detach().numpy().squeeze()

#%% Reconstruction
image3 = label2MaskMap([(pos_centre[0][0],pos_centre[0][1])], shape = (640,480), radius = 100, normalize = True)
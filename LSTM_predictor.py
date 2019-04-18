# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:24:56 2019

@author: talmezh
"""

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import imageio
import glob
import time
from random import shuffle

batch_size = 1
# %%
print('Loading data')
data_train = []
target_train = []
data_valid = []
target_valid = []
data_test = []
target_test = []
compteur = 0 
fileNameTarget = glob.glob("C:/Users/Denis/Desktop/LSTM_Input/LSTM_Target*.pt")
fileNameData= []
for im_path in glob.glob("C:/Users/Denis/Desktop/LSTM_Input/LSTM_Target*.pt"):
    dataStr = im_path[im_path.find('Target_') +7 :-1] + 't'
    fileNameData.append('C:/Users/Denis/Desktop/LSTM_Input/LSTM_Data_' + dataStr)
    if compteur > 3:
        x = torch.load(fileNameData[compteur-4]).float()
        y = torch.load(fileNameData[compteur-3]).float()
        z = torch.load(fileNameData[compteur-2]).float()
        w = torch.load(fileNameTarget[compteur-1]).float()
        if compteur < 100:
            target_train.append(w.view(1,1,300))
            data_train.append(torch.cat([x,y,z]).view(3,1,300))
        if compteur >= 100 and compteur < 122:
            target_valid.append(w.view(1,1,300))
            data_valid.append(torch.cat([x,y,z]).view(3,1,300))
        if compteur >= 122:
            target_test.append(w.view(1,1,300))
            data_test.append(torch.cat([x,y,z]).view(3,1,300))
    compteur += 1
print('Done loading data')
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
    
#%%
net = LSTM_predictor()

## %%
#data_train = []
#data_valid = []
#
#for i in range(50):
#    data_train.append(torch.randn(3,1,300))
#    if i%4 == 0:
#        data_valid.append(torch.randn(3,1,300))
#
#target_train = []
#target_valid = []    
#for i in range(50):
#    target_train.append(torch.randn(3,1,300))
#    if i%4 == 0:
#        target_valid.append(torch.randn(3,1,300))
# %%
    
def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    loss = 0
    dataSize = (len(train_loader))
    for batch_idx in range(int(dataSize)):
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(train_loader[batch_idx]), Variable(target_train[batch_idx])
        optimizer.zero_grad()
        output = model(data)  # calls the forward function

        loss += F.mse_loss(output, target)

        if batch_idx%batch_size==0:
            loss /= batch_size
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            loss = 0
    train_loss /= (dataSize / batch_size)
    return model, train_loss

def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    dataSize = (len(valid_loader))
    for batch_idx in range(dataSize):
        #data, target = Variable(valid_loader[batch_idx]).cuda(),Variable(target_valid[batch_idx]).cuda() # if you have access to a gpu
        data, target = Variable(valid_loader[batch_idx]), Variable(target_valid[batch_idx])
        output = model(data)
        loss = F.mse_loss(output, target)
        if (1-loss.item()*100)>0.98:
            correct += 1
        valid_loss += loss.item()  # sum up batch loss
#        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#        correct += pred.eq(target.data.view_as(pred).long()).sum() / 15 / 20

#valeurs = data[0]
#valeurs = target[0]
#image = valeurs.detach().numpy().squeeze()

    valid_loss /= dataSize
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, dataSize,
        100. * correct / dataSize))
    return 100. * correct / dataSize, valid_loss

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    dataSize = len(test_loader)
    for batch_idx in range(dataSize):
          #        data, target = Variable(valid_loader[batch_idx], volatile=True).cuda(),Variable(target_valid[batch_idx]).cuda() # if you have access to a gpu
        data, target = Variable(test_loader[batch_idx], volatile=True), Variable(target_test[batch_idx])
        output = model(data)
        loss = F.mse_loss(output, target)
        test_loss += loss.item()  # sum up batch loss
        if (1-loss.item()*100)>0.98:
            correct += 1
    test_loss /= dataSize
    print('\n' + "Test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, dataSize,
        100. * correct / dataSize))

def experiment(model, epochs=10, lr=0.001):
    best_precision = 0
    train_losses = []
    valid_losses = []
    valid_precision = []
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        print(epoch)
        if epoch == int(0.65*epochs):
            optimizer = optim.Adagrad(model.parameters(), lr=lr/10)
        model, train_loss = train(model, data_train, optimizer)
        train_losses.append(train_loss)
        precision, valid_loss = valid(model, data_valid)
        valid_losses.append(valid_loss)
        valid_precision.append(precision)
        if precision > best_precision:
            best_precision = precision
            best_model = model
        best_model = model
    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.plot(train_losses, 'b', valid_losses, 'm')
    ax1.set_title('Courbes d\'apprentissage: {}'.format(model.name))
    ax1.set_ylabel('MSE')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train', 'validation'])
    plt.figure(2)
    ax2 = plt.subplot(111)
    ax2.plot(valid_precision, 'k')
    ax2.set_title('Précision: {}'.format(model.name))
    ax2.set_ylabel('%')
    ax2.set_xlabel('Epoch')
    ax2.legend(['précision'])
    plt.show()
    return best_model, best_precision
        
# %%
best_precision = 0
for model in [LSTM_predictor()]:  # add your models in the list
    #    model.cuda()  # if you have access to a gpu
    model, precision = experiment(model, epochs=300, lr=0.001)
    if precision > best_precision:
        best_precision = precision
        best_model = model
        
test(model,data_test)
torch.save(model.state_dict(), 'best_model_LSTM.pth')


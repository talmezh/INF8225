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
data_train = []
data_valid = []

for i in range(50):
    data_train.append(torch.randn(3,1,300))
    if i%4 == 0:
        data_valid.append(torch.randn(3,1,300))

target_train = []
target_valid = []    
for i in range(50):
    target_train.append(torch.randn(3,1,300))
    if i%4 == 0:
        target_valid.append(torch.randn(3,1,300))
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
        maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target)
        valid_loss += loss.item()  # sum up batch loss
        if loss.item() <= 15:
            correct += 1;
        #pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #correct += pred.eq(target.data.view_as(pred).long()).cpu().sum() / 480 / 640

    valid_loss /= dataSize
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, dataSize,
        100. * correct / dataSize))
    return 100. * correct / dataSize, valid_loss


def experiment(model, epochs=10, lr=0.001):
    best_precision = 0
    train_losses = []
    valid_losses = []
    valid_precision = []
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
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
    model, precision = experiment(model, epochs=2, lr=0.01)
    if precision > best_precision:
        best_precision = precision
        best_model = model

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


# test = torch.from_numpy(np.array(imageio.imread('C:/Users/DenisCorbin/Desktop/CIL-1/Data/0001.png'))).view(1,1,480,640).double()
# conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double()
# result = conv(test)
batch_size = 23


# %%
class CNNEncoderDecoder(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoder, self).__init__()
        self.name = "Encoder Decoder CNN"
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True)

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        x = self.encoder(x)
        
        x = self.decoder(x)
        return x/x.max()


# %%
# model = CNNEncoderDecoder()
# result = model(data_train[0])
# epochs=10
# lr=0.001
# best_precision = 0
# train_losses = []
# valid_losses = []
# valid_precision = []
# optimizer = optim.Adam(model.parameters(), lr=lr)
# model, train_loss = train(model, data_train, optimizer)

# %%
print('Loading data')
data_train = []
target_train = []
data_valid = []
target_valid = []
compteur = 0
for im_path in glob.glob("C:/Users/DenisCorbin/Desktop/CIL1/Annotation/Output0/*.npy"):
    dataStr = im_path[im_path.find('\\') + 1:im_path.find('\\') + 5]
    im_pathData = 'C:/Users/DenisCorbin/Desktop/CIL1/Data/' + dataStr + '.png'
    if compteur % 5 == 0:
        target_valid.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_valid.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    else:
        target_train.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_train.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    compteur += 1
print('Done loading data')

#train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

#torch.cat(torch.stack(data_train),torch.stack(target_train))
#shuffle(target_train)
#target_train = torch.stack(target_train)


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
    print(LLE.item(), max_X_t, maxY_t, x_pred.item(), y_pred.item())
    return LLE


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
        #output_np = output.detach().numpy().squeeze()
        #target_np = target.detach().numpy().squeeze()
        #position = np.where(target_np == target_np.max())
        maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss += F.mse_loss(output, target) + LLE(output, maxY_t,max_X_t)
        #t = time.time()
        if batch_idx%batch_size==0:
            loss /= batch_size
            train_loss += loss.item()
            loss.backward()
            #print(time.time() - t)
            optimizer.step()
            loss = 0
    train_loss /= (dataSize / batch_size)
    # print('\n' + "train" + ' set: Average loss: {:.4f}\n'.format(train_loss))
    return model, train_loss


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    dataSize = (len(valid_loader))
    for batch_idx in range(dataSize):
        #        data, target = Variable(valid_loader[batch_idx], volatile=True).cuda(),Variable(target_valid[batch_idx]).cuda() # if you have access to a gpu
        data, target = Variable(valid_loader[batch_idx], volatile=True), Variable(target_valid[batch_idx])
        output = model(data)
        maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target) + LLE(output, maxY_t,max_X_t)
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


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        #        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    ax1.set_ylabel('LLE & MSE')
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
for model in [CNNEncoderDecoder()]:  # add your models in the list
    #    model.cuda()  # if you have access to a gpu
    model, precision = experiment(model, epochs=20, lr=0.01)
    if precision > best_precision:
        best_precision = precision
        best_model = model

valeurs = data_train[100]
image = valeurs.detach().numpy().squeeze()
valeurs = best_model(valeurs)
image = valeurs.detach().numpy().squeeze()

# test(best_model, test_loader)

valeurs = target_valid[0]
image = valeurs.detach().numpy().squeeze()
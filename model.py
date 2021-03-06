import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import imageio
import glob

# test = torch.from_numpy(np.array(imageio.imread('C:/Users/DenisCorbin/Desktop/CIL-1/Data/0001.png'))).view(1,1,480,640).double()
# conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double()
# result = conv(test)
batch_size = 10
save = 0
#typeTrain = 0
#typeValid = 0
#typeTest = 0
typeData = 0
typeTarget = 0
itteration_data = 1
itteration_target = 1
# %%
class CNNEncoderDecoder(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoder, self).__init__()
        self.itteration_data = itteration_data
        self.itteration_target = itteration_target
        self.type = ''
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
        if save:
#            if typeTrain:
#                self.type = 'Train_'
#            if typeValid:
#                self.type = 'Valid_'
#            if typeTest:
#                self.type = 'Test_'
            if typeTarget:
                torch.save(x, 'C:/Users/Denis/Desktop/LSTM_Input/LSTM_' +'Target_' + str(self.itteration_target).zfill(3) +'.pt')
                self.itteration_target += 1
            if typeData:
                torch.save(x, 'C:/Users/Denis/Desktop/LSTM_Input/LSTM_' +'Data_' + str(self.itteration_data).zfill(3) +'.pt')
                self.itteration_data += 1
        x = self.decoder(x)
        return x/x.max()

# %%
class CNNEncoderDecoderDropout(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoderDropout, self).__init__()
        self.itteration_data = itteration_data
        self.itteration_target = itteration_target
        self.type = ''
        self.name = "Encoder Decoder CNN with Dropout"
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(1).double(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)

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
class CNNEncoderDecoderMoreFeatures(nn.Module):
    def __init__(self):
        super(CNNEncoderDecoderMoreFeatures, self).__init__()
        self.itteration_data = itteration_data
        self.itteration_target = itteration_target
        self.type = ''
        self.name = "Encoder Decoder CNN with more features"
        self.enc_multi = [2, 2, 4, 4, 8, 16]
        self.enc_multi = [item * 8 for item in self.enc_multi]
        self.dec_multi = [16, 8, 4, 4, 2, 2]
        self.dec_multi = [item * 8 for item in self.dec_multi]
        print(self.enc_multi)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.enc_multi[0], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[0]).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.enc_multi[0], out_channels=self.enc_multi[1], kernel_size=3, stride=1,
                      padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[1]).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.enc_multi[1], out_channels=self.enc_multi[2], kernel_size=3, stride=1,
                      padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[2]).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.enc_multi[2], out_channels=self.enc_multi[3], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[3]).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.enc_multi[3], out_channels=self.enc_multi[4], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[4]).double(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=self.enc_multi[4], out_channels=self.enc_multi[5], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.enc_multi[5]).double(),
            nn.ReLU(inplace=True)

        )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.enc_multi[5], out_channels=self.dec_multi[0], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[0]).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.dec_multi[0], out_channels=self.dec_multi[1], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[1]).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.dec_multi[1], out_channels=self.dec_multi[2], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[2]).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.dec_multi[2], out_channels=self.dec_multi[3], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[3]).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.dec_multi[3], out_channels=self.dec_multi[4], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[4]).double(),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.dec_multi[4], out_channels=self.dec_multi[5], kernel_size=3, stride=1, padding=1).double(),
            nn.BatchNorm2d(self.dec_multi[5]).double(),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=self.dec_multi[5], out_channels=1, kernel_size=3, stride=1, padding=1).double(),
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

print('Loading data')
data_train = []
target_train = []
data_valid = []
target_valid = []
data_test = []
target_test = []
compteur = 0
for im_path in glob.glob("/home/talmezh/Documents/Data/CIL-01/Annotation/Output0/*.npy"):
    dataStr = im_path.split('/')[-1][:-4]
    im_pathData = '/home/talmezh/Documents/Data/CIL-01/Data/' + dataStr + '.png'
    if compteur < 100 :
        target_train.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_train.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    if compteur >= 100 and compteur < 122:
        target_valid.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_valid.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    if compteur >= 122:
        target_test.append(torch.from_numpy(np.load(im_path)).view(1, 1, 480, 640).double())
        data_test.append(torch.from_numpy(np.array(imageio.imread(im_pathData)) / 255).view(1, 1, 480, 640).double())
    compteur += 1
print('Done loading data')


def LLE(output, maxY_t,max_X_t):    
    pixel_sum = output.sum()
    
    #Calcul du x predit
    v_x = torch.arange(output.size(3)).double() #Vecteur allant de 0 a 639
    partial_sumx = torch.mv(output.squeeze(),v_x).sum() #Somme ponderee selon x
    x_pred = partial_sumx / pixel_sum
    
    output_T = torch.transpose(output,2,3) #Transposition de la matrice
    v_y = torch.arange(output.size(2)).double()
    partial_sumy = torch.mv(output_T.squeeze(),v_y).sum()

    y_pred = partial_sumy / pixel_sum

    LLE = torch.sqrt((torch.from_numpy(max_X_t).double() - x_pred).pow(2) + (torch.from_numpy(maxY_t).double() - y_pred).pow(2))
    return LLE


def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    # loss = 0
    dataSize = len(train_loader)
    for batch_idx in range(int(dataSize/batch_size)):
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        interval = [batch_idx * batch_size, (batch_idx + 1) * batch_size]
        data, target = train_loader[interval[0]:interval[1]], target_train[interval[0]:interval[1]]
        data = torch.stack(data).view(batch_size, 1, 480, 640).cuda()
        target = torch.stack(target).view(batch_size, 1, 480, 640).cuda()
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        #output_np = output.detach().numpy().squeeze()
        #target_np = target.detach().numpy().squeeze()
        #position = np.where(target_np == target_np.max())
        #maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target)# + LLE(output, maxY_t,max_X_t)
        #t = time.time():
        #loss /= batch_size
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        loss = 0
    train_loss /= (dataSize / batch_size)
    print('\n' + "train" + ' set: Average loss: {:.4f}\n'.format(train_loss))
    return model, train_loss


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    dataSize = len(valid_loader)
    for batch_idx in range(dataSize):
        data, target = valid_loader[batch_idx].cuda(), target_valid[batch_idx].cuda() # if you have access to a gpu
        output = model(data)
        #maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target)# + LLE(output, maxY_t,max_X_t)
        valid_loss += loss.item()  # sum up batch loss
        if loss.item() <= 15:
            correct += 1;
    valid_loss /= dataSize
    print('\n' + "Valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, dataSize,
        100. * correct / dataSize))
    return 100. * correct / dataSize, valid_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    dataSize = len(test_loader)
    for batch_idx in range(dataSize):
        data, target = test_loader[batch_idx].cuda(), target_test[batch_idx].cuda() # if you have access to a gpu
        output = model(data)
        #maxY_t, max_X_t = np.where(target.squeeze() == target.max())
        loss = F.mse_loss(output, target)# + LLE(output, maxY_t,max_X_t)
        test_loss += loss.item()  # sum up batch loss
        if loss.item() <= 15:
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
        # if epoch == int(0.75*epochs):
        #     optimizer = optim.Adagrad(model.parameters(), lr=lr/10)
        print(epoch)
        model, train_loss = train(model, data_train, optimizer)
        train_losses.append(train_loss)
        precision, valid_loss = valid(model, data_valid)
        valid_losses.append(valid_loss)
        valid_precision.append(precision)
        if precision >= best_precision:
            best_precision = precision
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
for model in [CNNEncoderDecoderMoreFeatures()]:  # add your models in the list
    model = model.cuda()  # if you have access to a gpu
    model, precision = experiment(model, epochs=3, lr=0.001)
test(model,data_test)


# #%% Saving intermediate values
# print('SAVING')
# save = 1
#
# # Train
# #typeTrain = 1
# for i in range(len(data_train)):
#     typeTarget = 1
#     model(target_train[i])
#     typeTarget = 0
#     typeData = 1
#     model(data_train[i])
#     typeData = 0
# #typeTrain = 0
#
# # Valid
# #typeValid = 1
# for i in range(len(data_valid)):
#     typeTarget = 1
#     model(target_valid[i])
#     typeTarget = 0
#     typeData = 1
#     model(data_valid[i])
#     typeData = 0
# #typeValid = 0
#
# # Test
# #typeTest = 1
# for i in range(len(data_test)):
#     typeTarget = 1
#     model(target_test[i])
#     typeTarget = 0
#     typeData = 1
#     model(data_test[i])
#     typeData = 0
# #typeTest= 0
# save = 0
# print('DONE SAVING')
# #%% Saving model
# #torch.save(model.state_dict(), 'best_ED_300_mb1_morefeatures.pth')
#
#
# #%% Loading model
# model = CNNEncoderDecoder()
# model.load_state_dict(torch.load('C:/Users/DenisCorbin/Documents/GitHub/INF8225/best_ED_300_mb5.pth'))
# model.eval()
# test(model,data_test)


#%%
#valeurs = data_train[100]
#image = valeurs.detach().numpy().squeeze()
#valeurs = best_model(valeurs)
#image = valeurs.detach().numpy().squeeze()
#valeurs = target_valid[0]
#image = valeurs.detach().numpy().squeeze()
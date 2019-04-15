import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch import optim

from torchvision.datasets.mnist import MNIST


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
            ]


train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)

train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

mask = np.ones(60000)
mask[train_idx] = 0

valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]

batch_size = 200
test_batch_size = batch_size

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_data,
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)
    
#%%    
    
class LogisticReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Régression logistique"
        self.fc1 = nn.Linear(28*28,10)
        
    def forward(self,sample):
        sample = sample.view(sample.size()[0],-1)
        pred = F.log_softmax(self.fc1(sample),dim=1)
        return pred

#%%    
    
class FullyCon(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Fully connected"
        self.fc1 = nn.Linear(28*28,1000)
        self.fc2 = nn.Linear(1000,800)
        self.fc3 = nn.Linear(800,400)
        self.fc4 = nn.Linear(400,100)
        self.fc5 = nn.Linear(100,10)

    def forward(self,sample):
        sample = sample.view(sample.size()[0],-1)
        sample = F.relu(self.fc1(sample))
        sample = F.relu(self.fc2(sample))
        sample = F.relu(self.fc3(sample))
        sample = F.relu(self.fc4(sample))
        pred = F.log_softmax(self.fc5(sample),dim=1)
        return pred
    
#%%
class BlitzNet(nn.Module):

    def __init__(self):
        super(BlitzNet, self).__init__()
        self.name = "BlitzNet"
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#%%
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

#%%

def train(model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        #print(batch_idx)
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        #print(loss.item())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= (len(train_loader.dataset)/batch_size)
    print('\n' + "train" + ' set: Average loss: {:.4f}\n'.format(train_loss))
    return model, train_loss


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    for data, target in valid_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        valid_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    return 100. *correct / len(valid_loader.dataset), valid_loss

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model, train_loss = train(model, train_loader, optimizer)
        train_losses.append(train_loss)
        precision, valid_loss = valid(model, valid_loader)
        valid_losses.append(valid_loss)
        valid_precision.append(precision)
        if precision.item() > best_precision:
            best_precision = precision
            best_model = model
    plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.plot(train_losses, 'b', valid_losses, 'm')
    ax1.set_title('Courbes d\'apprentissage: {}'.format(model.name))
    ax1.set_ylabel('Log négatif de vraisemblance moyenne')
    ax1.set_xlabel('Epoch')
    ax1.legend(['train','validation'])
    plt.figure(2)
    ax2 = plt.subplot(111)
    ax2.plot(valid_precision, 'k')
    ax2.set_title('Précision: {}'.format(model.name))
    ax2.set_ylabel('%')
    ax2.set_xlabel('Epoch')
    ax2.legend(['précision'])
    plt.show()
    return best_model, best_precision


#%%
best_precision = 0
for model in [BlitzNet()]:  # add your models in the list
    model.cuda()  # if you have access to a gpu
    model, precision = experiment(model, epochs = 10, lr=0.1)
    if precision > best_precision:
        best_precision = precision
        best_model = model

test(best_model, test_loader)
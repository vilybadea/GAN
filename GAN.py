'''MiniMax Theorem

Maximin:
    Largest value player A can gain not knowing the actions of player B

Minimax: 
    Smallest value player V can force player A to receive without knowing his actions

Discriminator:
    Output 1 if real image, 0 if fake image. 
    Gets half real and half fake images as input.
    Wants to maximize stochastic gradient function.

    Input layer must be an image in case of ConvNet
    Output must then be a 1 class sigmoid output. (last layer should be fully connected)

Generator: 
    Take noise (can be Gaussian) and generate image. 
    Wants to minimize stochastic gradient function.

    Input layer is same size of discriminator's input, as noise.
    Last layer should be an image in case of ConvNet

Nash equilibrium is when discriminator outputs 0.5
'''
import torch
import torch.optim as optim 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
import torchvision.transforms as transforms 
from tqdm import tqdm

#mini-batch size
mb_size = 64

#this will transform data to tensor format which is pytorch's expected format
transform = transforms.Compose([transforms.ToTensor()])

#here we download the dataset and transform it, train=True will only download training dataset
trainset = torchvision.datasets.MNIST(root = './NewData', download = True, train = True, transform = transform)

#now we want to make a loader that we can iterate through and load
#our data one mini batch at a time
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=mb_size)

#just for example, we are going to visualize
#we define an iterator
data_iter = iter(trainloader)

#getting the next batch of the images and labels
images, labels = data_iter.next()
test = images.view(images.size(0), -1)
#print(test.size())

Z_dim = 100
X_dim = test.size(1)
h_dim = 128
lr = 1e-3


def imshow(img):
    im = torchvision.utils.make_grid(img)
    npimg = im.numpy()
    print(npimg.shape)
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()
'''
imshow(images)
'''

#initialize weights for generator and discriminator
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

#class for generator, inheriting nn.Module class 
class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()

        self.model = nn.Sequential(
            
            nn.Linear(Z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
            
        )
        self.model.apply(init_weight)

    def forward(self, input):
        return self.model(input)


class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
    
        self.model = nn.Sequential(
            
            nn.Linear(X_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
            
        )
        self.model.apply(init_weight)
    def forward(self, input):
        return self.model(input)    

G = Gen()
D = Dis()

G_solver = optim.Adam(G.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)


for epoch in tqdm(range(8)):
    G_loss_run = 0.0
    D_loss_run = 0.0
    for i, data in tqdm(enumerate(trainloader)):
        X,_ = data 
        mb_size = X.size(0)
        X = X.view(X.size(0), -1)

        one_labels = torch.ones(mb_size, 1)
        zero_labels = torch.zeros(mb_size, 1)

        z = torch.randn(mb_size, Z_dim)
        G_sample = G(z)
        D_fake = D(G_sample)
        D_real = D(X)

        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        print("Discriminator Fake Loss: ", D_fake_loss)
        print("Discriminator Real Loss: ", D_real_loss)

        D_loss = D_real_loss + D_fake_loss
        D_solver.zero_grad()
        D_loss.backward()
        D_solver.step()

        z = torch.randn(mb_size, Z_dim)
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = F.binary_cross_entropy(D_fake, one_labels)
        G_solver.zero_grad()
        G_loss.backward()
        G_solver.step()

        #samples = G(z).detach()
        #samples = samples.view(mb_size, 1, 28, 28)
        #imshow(samples)
        #print('Epoch: {}', 'G_loss: {}', 'D_loss: {}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
    samples = G(z).detach()
    samples = samples.view(mb_size, 1, 28, 28)
    imshow(samples)



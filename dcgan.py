# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#%%
# Root directory for dataset
dataroot = "c:/users/ramin/desktop/imgs/"
workers = 2
batch_size = 128
image_size = 64
num_epochs = 5
ngpu = 1
device = 'cuda'


dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


nz = 100
ngf = 64
ndf = 64
nc = 3
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False), # 64,512
            nn.BatchNorm2d(ngf * 8), #512
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), # 512,256
            nn.BatchNorm2d(ngf * 4), # 256
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False), # 256,128
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False), # 128,64
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False), #64,64
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


netG = Generator(ngpu).to(device)
if (device == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)


nc = 3
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), # 3,64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # 64,128
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 128,256
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), # 256,512
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # 512,1
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

netD = Discriminator(ngpu).to(device)
if (device == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)


nz = 100
lr = 0.0002
beta1 = 0.5
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#%%
real_label = 1.
fake_label = 0.
img_list = []
G_losses = []
D_losses = []
iters = 0

import sys

for epoch in range(20):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device) # torch.Size([128, 3, 64, 64])
        b_size = real_cpu.size(0) # 128
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # torch.Size([128]) => fill with 1
        output = netD(real_cpu).view(-1) # 128,1,1,1 => 128
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item() # mean of real output
        noise = torch.randn(b_size, nz, 1, 1, device=device) # torch.Size([128, 100, 1, 1])
        fake = netG(noise) # torch.Size([128, 3, 64, 64])
        label.fill_(fake_label) # torch.Size([128]) => fill with 0
        output = netD(fake.detach()).view(-1) # 128,1,1,1 => 128
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item() # mean of fake output
        errD = errD_real + errD_fake # loss real + loss fake
        optimizerD.step()
        netG.zero_grad()
        label.fill_(real_label) # (batch, ) => fill with 1
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item() # mean of fake output
        optimizerG.step()
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1        
        
#%%
from torchvision import transforms as T
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
netG.eval()
with torch.no_grad():    
    fake1 = netG(fixed_noise).detach().cpu()[0]
from PIL import Image
T.ToPILImage()(fake1.to('cpu'))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:28:27 2019

@author: gaoyi
"""

import torch
import itertools
import dataset
from model import vae
import prior 
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#%%params
batch = 64

lr = 1e-4
epochs = 100
#%%

#defiine model
encoder = vae.Encoder().cuda()
decoder = vae.Decoder().cuda()
discriminator = vae.Discriminator().cuda()

#define loss
adversarial_loss = torch.nn.BCELoss().cuda()
pixelwise_loss = torch.nn.MSELoss().cuda()
#pixelwise_loss = torch.nn.L1Loss().cuda()
#pixelwise_loss = torch.nn.SmoothL1Loss().cuda()
#load data
train_loader,train_loader2, devel_loader, test_loader, traindevel_loader = dataset.preprocess(batch_size=batch)

optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()),lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor
#%%

def plot(mode):
    
    
    latent=[]
    latent_y=[]
    if mode!='test':
        if mode=='devel':
            loader = devel_loader
        elif mode =='train':
            loader = train_loader2
        
        
        for (x,y) in loader:
            latent.append(encoder(x.cuda()).cpu().detach().numpy())
            latent_y.append(y.detach().numpy())
            
        latent = np.concatenate(latent)   
        latent_y = np.concatenate(latent_y)
        
        
        plt.figure()
        #['Canonical','Crying','Junk','Laughing','Non-canonical']
        x0 = plt.scatter(latent[latent_y==0][:,0],latent[latent_y==0][:,1],s=5,c='red', alpha = 0.5,label='0')
        x1 = plt.scatter(latent[latent_y==1][:,0],latent[latent_y==1][:,1],s=5,c='blue', alpha = 0.5,label='1')
        x2 = plt.scatter(latent[latent_y==2][:,0],latent[latent_y==2][:,1],s=5,c='lightgreen', alpha = 0.5,label='2')
        x3 = plt.scatter(latent[latent_y==3][:,0],latent[latent_y==3][:,1],s=5,c='gold', alpha = 0.5,label='3')
        x4 = plt.scatter(latent[latent_y==4][:,0],latent[latent_y==4][:,1],s=5,c='m', alpha = 0.5,label='4')       
        plt.legend(handles=[x0,x1,x2,x3,x4])
        plt.ylim((-5, 5))
        plt.xlim((-5, 5))
        plt.show()
    
    else:
        
        loader = test_loader
        for x in loader:
            latent.append(encoder(x[0].cuda()).cpu().detach().numpy())
            
        latent = np.concatenate(latent)   
        
        plt.figure()
        x0 = plt.scatter(latent[:,0],latent[:,1],s=5,c='red', alpha = 0.5,label='0')

        plt.legend(handles=[x0])
        plt.ylim((-5, 5))
        plt.xlim((-5, 5))
        plt.show()
   

    return 0



def loss_plot(loss_log):
    iters = range(len(epoch_log['d_loss']))
    plt.figure()

    # loss
    plt.plot(iters, epoch_log['d_loss'], 'b', label='d loss')
    plt.plot(iters, epoch_log['e_loss'], 'r', label='g loss')
    plt.plot(iters, epoch_log['g_loss'], 'g', label='rec loss')


    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="upper right")
    plt.show()

#%%train
epoch_log = {'d_loss':[], 'g_loss':[],'e_loss':[]}
for epoch in range(epochs):
    log = {'d_loss':[], 'g_loss':[],'e_loss':[]}
    for i, (imgs,labs) in enumerate(train_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        
        x_id_one_hot_vector = torch.zeros((len(imgs), 5)).cuda()
        x_id_one_hot_vector[np.arange(len(imgs)), labs] = 1
        encoded_imgs = torch.cat((encoded_imgs,x_id_one_hot_vector),1)
        
        # Loss measures generator's ability to fool the discriminator
        e_loss =  adversarial_loss(discriminator(encoded_imgs), valid)
        g_loss =  pixelwise_loss(decoded_imgs, real_imgs)
        
        r_loss = e_loss + g_loss

        r_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
#            z = Variable(Tensor(np.random.normal(0, 20, (imgs.shape[0], opt.latent_dim))))
        z_id_ = np.random.randint(0, 5, size=[len(imgs)])
        z = Variable(Tensor(prior.gaussian_mixture2(len(imgs), 2,n_labels=5, label_indices=z_id_))).cuda()
        
        z_id_one_hot_vector = torch.zeros((len(imgs), 5)).cuda()
        z_id_one_hot_vector[np.arange(len(imgs)), z_id_] = 1
        
        
        z = torch.cat((z,z_id_one_hot_vector),1)
        
        

        
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
 
        d_loss.backward()
        optimizer_D.step()

#        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(train_loader),
#                                                            d_loss.item(), r_loss.item()))
        
        log['d_loss'].append(d_loss.item())
        log['e_loss'].append(e_loss.item())
        log['g_loss'].append(g_loss.item())
    
    epoch_log['d_loss'].append(np.mean(log['d_loss']))
    epoch_log['e_loss'].append(np.mean(log['e_loss']))
    epoch_log['g_loss'].append(np.mean(log['g_loss']))
    print ("[Epoch %d/%d]"% (epoch,epochs))
    if (epoch+1)%10==0:
        encoder = encoder.eval()
        
        print('train')
        _ = plot('train')
        print('devel')
        _ = plot('devel')
        print('test')
        _ = plot('test')
        encoder = encoder.train()
        
        
        
loss_plot(epoch_log)
torch.save(decoder.cpu().state_dict(), 'decoder_params_64_Train_ACD.pkl')  
#ls.append(reduce(X_train,X_test,encoder))




    

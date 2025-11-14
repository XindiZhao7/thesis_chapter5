#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import argparse
import pickle
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tools.data import get_dataloaders
from sklearn.preprocessing import StandardScaler
from icp import pValues, calculate_q
from models.MLP import MLP1, MLP2, BMLP1, BMLP2
from openpyxl import Workbook


# In[2]:



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# In[3]:


def main():
    parser = argparse.ArgumentParser(description='CP pruning')
    parser.add_argument('--model', default='mlp1', type=str,
                        help='model selection, choices: mlp1, mlp2, bmlp1, bmlp2, vgg16, vgg19, resnet18, resnet34',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])
    parser.add_argument('--dataset', default='not_mnist', type=str,
                        help='dataset selection, choices: mnist, fashion_mnist, covtype, svhn, tmnist, cifar10, cifar100',
                        choices=['mnist', 'fashion_mnist', 'covtype', 'svhn', 'tmnist', 'cifar10', 'cifar100'])
    parser.add_argument('--feature-dim', type=tuple, default=(784,),
                        help='feature dimension (default: 10)')       
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes (default: 10)')    
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--patience', type=int, default=20,
                        help='number of epochs for early stopping (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.000001,
                        help='weight decay (default: 0.0)')    
    parser.add_argument('--epsilon', type=int, default=0.01,
                        help='significance level (default: 0.01)')
    parser.add_argument('--save-path', default='results/nmnist-mlp1-t0.pt', type=str,
                        help='save path for model weights')    
    parser.add_argument('--method', default='snr', type=str,
                        help='pruning criterion selection, choices: abs-cp, sign-cp, magnitude, taylor, snr',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])

    args = parser.parse_args([])
    if args.model=='mlp1':
        from tools.vanilla_training import train
        model = MLP1(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [500, 300]
    elif args.model=='mlp2':
        from tools.vanilla_training import train
        model = MLP2(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [1024, 1024, 512]
    elif args.model=='bmlp1':
        from tools.variational_inference import train
        model = BMLP1(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [500, 300]
    elif args.model=='bmlp2':
        from tools.variational_inference import train
        model = BMLP2(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [1024, 1024, 512]
        
    num_hidden_layers = len(hidden_units)
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    train_loader, calib_loader, test_loader, valid_loader = get_dataloaders(args.dataset, args.batch_size)
    
               
    model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, None, device, finetune=False)

    
    file = open("results/train_time.txt", "w")
    file.write(str(train_time))
    file.close()
    
    file = open("results/train_epoch.txt", "w")
    file.write(str(epoch))
    file.close()
    
    return


# In[4]:


main()


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pickle
import torch
import numpy as np
from torch.utils.data import sampler, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


# In[ ]:


def get_dataset(dataset):

    if dataset=='mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])        
        train_data = datasets.MNIST(root='data',
                                    train=True,
                                    download=True, 
                                    transform=transform)

        test_data = datasets.MNIST(root='data',
                                   train=False,
                                   download=True,
                                   transform=transform)      
    if dataset=='fashion_mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_data = datasets.FashionMNIST(root="data/fashion_data",
                                           train=True,
                                           download=True, 
                                           transform=transform)

        test_data = datasets.FashionMNIST(root="data/fashion_data",
                                          train=False,
                                          download=True,
                                          transform=transform)

    return train_data, test_data


# In[ ]:


def get_dataloaders(dataset, batch_size):

    # choose the training and test datasets
    #train_data, test_data = get_dataset(dataset)

    # obtain training indices that will be used for validation
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        train_data, test_data = get_dataset(dataset)
        file = open(dataset, 'rb')
        permutations = pickle.load(file)
        indices = permutations['0']
        train_idx, valid_idx, calib_idx= indices[:50000], indices[50000:55000], indices[55000:]
    elif dataset == 'not_mnist':
        datasets = np.load('notmnist.npy', allow_pickle=True).tolist()
        train_data, test_data = datasets['train'], datasets['test']
        file = open('mnist', 'rb')
        permutations = pickle.load(file)
        indices = permutations['0']
        train_idx, valid_idx, calib_idx= indices[:50000], indices[50000:55000], indices[55000:]
        train_images, train_labels = train_data['x'][train_idx], train_data['y'][train_idx]
        cal_images, cal_labels = train_data['x'][calib_idx], train_data['y'][calib_idx]
        val_images, val_labels = train_data['x'][valid_idx], train_data['y'][valid_idx]
        test_images, test_labels = test_data['x'], test_data['y']
        
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_images).to(torch.float32),
                                                        torch.from_numpy(train_labels).to(torch.long))
        cal_dataset = torch.utils.data.TensorDataset(torch.from_numpy(cal_images).to(torch.float32),
                                                        torch.from_numpy(cal_labels).to(torch.long))
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_images).to(torch.float32),
                                                        torch.from_numpy(val_labels).to(torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_images).to(torch.float32),
                                                        torch.from_numpy(test_labels).to(torch.long))        
    elif dataset == 'covtype':
        datasets = np.load('covtype.npy', allow_pickle=True).tolist()
        
        X, y = datasets['X'], datasets['y']
        
        trial = datasets['t1']
        
        train_indices, test_indices = trial['train_indices'], trial['test_indices']
        
        train_images, train_labels = X[train_indices[:36000]], y[train_indices[:36000]]
        cal_images, cal_labels = X[train_indices[36000:42000]], y[train_indices[36000:42000]]
        val_images, val_labels = X[train_indices[42000:]], y[train_indices[42000:]]
        test_images, test_labels = X[test_indices], y[test_indices]
    
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_images).to(torch.float32),
                                                        torch.from_numpy(train_labels).to(torch.long))
        cal_dataset = torch.utils.data.TensorDataset(torch.from_numpy(cal_images).to(torch.float32),
                                                        torch.from_numpy(cal_labels).to(torch.long))
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_images).to(torch.float32),
                                                        torch.from_numpy(val_labels).to(torch.long))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_images).to(torch.float32),
                                                        torch.from_numpy(test_labels).to(torch.long))
                                                                      
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    calib_loader = DataLoader(cal_dataset, batch_size=len(cal_labels), shuffle=False)
    valid_loader = DataLoader(val_dataset, batch_size=len(val_labels), shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=len(test_labels), shuffle=False)
    
    return train_loader, calib_loader, test_loader, valid_loader


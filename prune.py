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



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


# In[ ]:


def brute_force_mask(model, node_index):
    masks = list()
    nunits_by_layer = list()
    
    for name, param in model.named_parameters():
        if 'gate' in name:
            masks.append(param)
        elif 'out' in name:
            break
        else:
            if len(param.shape)>1:
                nunits = param.shape[0]
                nunits_by_layer.append(nunits)

    flatten_mask = torch.cat(masks, dim=0)

    new_mask = flatten_mask.clone()    
    new_mask[node_index]=0
    new_masks=torch.split(new_mask, nunits_by_layer)

    return new_masks


def updatemodel(model, init_weights, mask):
    gate_counter = 0
    for name, param in model.named_parameters():
        if 'gate' in name:
            param.data = mask[gate_counter]
            gate_counter += 1
        else:
            if init_weights != None:
                param.data = init_weights[name]

    return model


def reset_mask(model):
    for name, param in model.named_parameters():
        if 'gate' in name:
            param.data = torch.ones(param.shape, requires_grad=False).to(device)
    
    return model


def getmask(model, flatten_importances, percentile = 0.2, printvalue=True):
    masks = list()
    nunits_by_layer = list()
    
    for name, param in model.named_parameters():
        if 'gate' in name:
            masks.append(param)
            nunits_by_layer.append(len(param))
        elif 'out' in name:
            break


    flatten_mask = torch.cat(masks, dim=0)
    
    remaining_nodes_num = int(np.sum(flatten_mask.cpu().numpy()))
    remaining_nodes_indices = np.where(flatten_mask.cpu().numpy()==1)[0]
    
    _, indices = torch.sort(flatten_importances[remaining_nodes_indices])
    
    new_mask = flatten_mask.clone()    
    new_mask[remaining_nodes_indices[indices.cpu().numpy()][:int(percentile*len(indices))]]=0
    new_masks=torch.split(new_mask, nunits_by_layer)
    
    percent = torch.sum(new_mask)/len(flatten_mask)
    

    remaining_nodes = [torch.sum(mask).item() for mask in new_masks]
    print('Percentage remaining', percent.cpu().numpy(), end = ' ')
    print('Layer nodes:', remaining_nodes, end = ' ')
    print('\n')

    file = open("results/"+str(percent.cpu().numpy())+".txt", "w")
    file.write(str(remaining_nodes)+"\n")
    file.close()
    
    return new_masks,  percent
# In[ ]:



# In[ ]:


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
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='whehter there exists a pretrained model')        
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of classes (default: 10)')    
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--finetune-epochs', type=int, default=13,
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
    parser.add_argument('--method', default='abs-cp', type=str,
                        help='pruning criterion selection, choices: abs-cp, sign-cp, magnitude, taylor, snr',
                        choices=['abs-cp', 'sign-cp', 'magnitude', 'taylor', 'snr'])


    
    args = parser.parse_args([])

    percent = 1.0
    mask = None
    importances = list()
    mean_sizes = list()
    coverages = list()
    accuracys = list()
    timer={'train':[], 'importance_estimation':[], 'finetune':[], 'train_epoch':[], 'finetune_epoch':[]}
    performance={'mean_size':[], 'coverage':[], 'acc':[]}

    if args.model=='mlp1':
        from tools.vanilla_training import train
        from tools.test_model import evaluate
        model = MLP1(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [500, 300]
    elif args.model=='mlp2':
        from tools.vanilla_training import train
        from tools.test_model import evaluate
        model = MLP2(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [1024, 1024, 512]
    elif args.model=='bmlp1':
        from tools.variational_inference import train
        from tools.test_model import evaluate_ensemble
        model = BMLP1(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [500, 300]
    elif args.model=='bmlp2':
        from tools.variational_inference import train
        from tools.test_model import evaluate_ensemble
        model = BMLP2(args.feature_dim[0], args.num_classes).to(device)
        hidden_units = [1024, 1024, 512]
    num_hidden_layers = len(hidden_units)
    num_hidden_units = np.sum(hidden_units)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    train_loader, calib_loader, test_loader, valid_loader = get_dataloaders(args.dataset, args.batch_size)
    
    if args.pretrained == True:
        model.load_state_dict(torch.load(args.save_path))
        f = open("results/train_time.txt")
        train_time = f.read()
        f.close()
    else:
        # train model                  
        model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, None, device, finetune=False)
        args.finetune_epochs = epoch - args.patience
    
    timer['train'].append(train_time)
    timer['train_epoch'].append(args.finetune_epochs+args.patience)
    
    mean_size, coverage, acc = evaluate(model, calib_loader, valid_loader, args.epsilon, device)
    mean_sizes.append(mean_size)
    coverages.append(coverage)
    accuracys.append(acc)

    mean_size, coverage, acc = evaluate(model, calib_loader, test_loader, args.epsilon, device)
    performance['mean_size'].append(mean_size)
    performance['coverage'].append(coverage)
    performance['acc'].append(acc)
    
    
    if args.method == 'abs-cp' or args.method == 'sign-cp':
        start=time.time()
        for i in range(num_hidden_units):
            model=reset_mask(model)
            mask=brute_force_mask(model, i)
            model=updatemodel(model, None, mask)
            mean_size, coverage, acc = evaluate(model, calib_loader, valid_loader, args.epsilon, device)
            mean_sizes.append(mean_size)
            coverages.append(coverage)
            accuracys.append(acc)
        end=time.time()
        
        np.save('avgsize.npy', mean_sizes)
        
        if args.method == 'abs-cp':
            node_importance = np.abs(np.array(mean_sizes[1:])-mean_sizes[0])
        if args.method == 'sign-cp':
            node_importance = np.array(mean_sizes[1:])-mean_sizes[0]


    elif args.method == 'magnitude':
        start=time.time()
        for name, param in model.named_parameters():
            if 'gate' in name or 'out' in name:
                continue
            else:
                if len(param.shape)>1:
                    importance = torch.abs(param).view(param.shape[0],-1).sum(dim=1)
                    importances.append(importance)
        end=time.time()
        node_importance = torch.cat(importances, dim=0).cpu().detach().numpy()
    elif args.method == 'snr':
        start=time.time()
        omega1=(model.fc1.weight.mu/model.fc1.weight.sigma).sum(dim=1)
        omega2=(model.fc2.weight.mu/model.fc2.weight.sigma).sum(dim=1)
        omega3=(model.fc3.weight.mu/model.fc3.weight.sigma).sum(dim=1)
        end=time.time()        
        node_importance = torch.cat([omega1, omega2, omega3], dim=0).cpu().detach().numpy()

    else:
        print("Invalid method.")

    timer['importance_estimation'].append(end-start)
    
    np.save('importance.npy',node_importance)
    
    print('\n>>> Start Pruning<<<')
    
    while percent > 0.01:
        last_percent = percent
        mask, percent = getmask(model, torch.tensor(node_importance))
        if last_percent == percent:
            break
        initial_weights = torch.load(args.save_path)
        model=updatemodel(model, initial_weights, mask)
        print('**************************************************************')
        print('\n>>> Retraining<<<')
        model, train_time, epoch = train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, mask, device, finetune=True)
        timer['finetune'].append(train_time)
        timer['finetune_epoch'].append(epoch)
        mean_size, coverage, acc=evaluate(model, calib_loader, test_loader, args.epsilon, device)
        performance['mean_size'].append(mean_size)
        performance['coverage'].append(coverage)
        performance['acc'].append(acc)
        print('**************************************************************')

    
    np.save('performance.npy', performance)
    np.save('time.npy', timer)
    return


# In[2]:


main()


# In[ ]:





# In[ ]:





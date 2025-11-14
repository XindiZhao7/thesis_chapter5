#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
import numpy as np
import torch.nn as nn
from test_model import evaluate


# In[ ]:


def maskgradient(model, mask, size_mask):
    gate_counter = 0   
    for name, param in model.named_parameters():
        if param.requires_grad is not False and 'fc' not in name and 'bn' not in name:
            param.grad*=mask[gate_counter].view(*size_mask[gate_counter])
            gate_counter += 1


# In[ ]:


def train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, mask, device, finetune=False):
    eps = args.epsilon
    patience = args.patience
    batch_size = args.batch_size
    best_mean_size = args.num_classes
    max_mean_size = args.num_classes
    size_mask=[[-1, 1, 1, 1]]*num_hidden_layers
    if not finetune:
        nepochs = args.epochs
        save_path = args.save_path
    else:
        nepochs = args.finetune_epochs
        save_path = 'checkpoint.pt'

    cat_batch_size = int(0.5*batch_size)
    t = 0

    for epoch in range(1, nepochs + 1):
        train_running_loss = []
        model.train() # prep model for training
        start = time.time()
        for batch, (data, target) in enumerate(train_loader, 1):
            idx = np.random.permutation(batch_size)
            Xcat, ycat = data[idx[:cat_batch_size]].to(device), target[idx[:cat_batch_size]].to(device)
            Xtr, ytr = data[idx[cat_batch_size:]].to(device), target[idx[cat_batch_size:]].to(device)
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out_cat = model(Xcat)
            cat_ncs = nn.LogSoftmax(dim=1)(out_cat)
            catAlphas = cat_ncs[[torch.arange(0, len(ycat)), ycat]]
            q = calculate_q(catAlphas, eps)
            out = model(Xtr)
            ncs = nn.LogSoftmax(dim=1)(out)
            # calculate the loss
            loss = criterion(ncs, q)
            # backward pass: compute gradient of the loss with respect to model parameters
            train_running_loss.append(loss.item())
            optimizer.zero_grad() 
            loss.backward()
            # perform a single optimization step (parameter update)
            if mask is not None:
                maskgradient(model, mask, size_mask)
            optimizer.step()
        end = time.time()
        t += (end-start)
  
        # validate the model
        mean_size, coverage, acc=evaluate(model, calib_loader, valid_loader, eps, device)
        # early stop
        if mean_size >= max_mean_size:
            break
        if mean_size < best_mean_size:
            counter = 0
            best_mean_size = mean_size
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1

        if counter >= patience:
            print("Early Stop!")
            break
        train_epoch_loss = np.mean(train_running_loss)
        print(f"Epoch: {epoch:d}, Training loss: {train_epoch_loss:.3f}, training acc: {acc:.3f}")            
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(save_path))
    
    return  model, t, epoch


# In[ ]:


def evaluate(model, calib_loader, test_loader, eps, device):
    correct = 0
    model.eval()
    with torch.no_grad():
        for X_cal, y_cal in calib_loader:
            X_cal, y_cal = X_cal.to(device), y_cal.to(device)
            output = model(X_cal)
            cal_ncs = output.softmax(dim=1)
            calAlphas = cal_ncs[[torch.arange(0, len(y_cal)), y_cal]]

        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            output = model(X_test)
            test_ncs = output.softmax(dim=1)
            pvals = pValues(calAlphas, test_ncs, randomized=False)
            
            index = np.expand_dims(np.argwhere(pvals>eps), axis=2).tolist()
            prediction = torch.zeros_like(test_ncs)
            for i in index:
                prediction[i] = 1
            print('**************************************************************')
            mean_size = len(index)/len(y_test)
            print('Ineff: %f' %mean_size)

            correct_ = np.array([prediction[i, y_test[i]] for i in range(len(y_test))])
            n_correct = len(np.where(correct_==1)[0])
            coverage = n_correct/len(y_test)
            print('Err: %.1f' %(100*coverage))
            
            y_hat = torch.argmax(test_ncs, axis=1)
            acc = torch.eq(y_hat, y_test).float().sum().item()/y_test.size(0)
            print('Acc: %f' % (acc*100))
            
    return mean_size, coverage, acc


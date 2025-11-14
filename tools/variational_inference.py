#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import torch
import numpy as np
import torch.nn as nn
from tools.test_model import evaluate_ensemble


# In[ ]:


def maskgradient(model, mask, size_mask):
    for name, param in model.named_parameters():
        if 'fc1' in name:
            param.grad*=mask[0].view(*size_mask[0])
        elif 'fc2' in name:
            param.grad*=mask[1].view(*size_mask[1])
        elif 'fc3' in name:
            param.grad*=mask[2].view(*size_mask[2])            
        else:
            continue
# In[ ]:


def train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, mask, device, finetune=False):
    eps = args.epsilon
    patience = args.patience    
    best_mean_size = args.num_classes
    max_mean_size = args.num_classes
    size_mask=[[-1,1]]*num_hidden_layers
    num_batches = 36000//args.batch_size
    if not finetune:
        nepochs = args.epochs
        save_path = args.save_path
    else:
        nepochs = args.finetune_epochs
        save_path = 'checkpoint.pt'    
    
    t=0
    for epoch in range(1, nepochs + 1):
        train_running_loss = []
        model.train()
        start = time.time()
        for batch, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(data, target, args.num_classes, args.batch_size, num_batches)

            if not finetune:
                # Compute L1 loss component
                loss = loss + args.weight_decay * sum(torch.abs(p).sum() for p in model.parameters())

            train_running_loss.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()
            if mask is not None:
                maskgradient(model, mask, size_mask)
            optimizer.step()
        end = time.time()
        t += (end-start)
        
        mean_size, coverage, acc=evaluate_ensemble(model, calib_loader, valid_loader, eps, 1, args.num_classes, True)
        
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


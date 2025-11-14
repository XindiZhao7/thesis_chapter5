#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import torch
import numpy as np
import torch.nn as nn
from tools.test_model import evaluate


# In[ ]:
def maskgradient(model, mask, size_mask):
    gate_counter = 0   
    for name, param in model.named_parameters():
        if param.requires_grad is not False and 'out' not in name:
            param.grad*=mask[gate_counter].view(*size_mask[gate_counter])
            gate_counter += 1


def train(args, model, train_loader, calib_loader, valid_loader, criterion, optimizer, num_hidden_layers, mask, device, finetune=False):   
    t = 0
    eps = args.epsilon
    patience = args.patience    
    best_mean_size = args.num_classes
    max_mean_size = args.num_classes
    size_mask=[[-1, 1]]*num_hidden_layers
    if not finetune:
        nepochs = args.epochs
        save_path = args.save_path
    else:
        nepochs = args.finetune_epochs
        save_path = 'checkpoint.pt'
    
    for epoch in range(1, nepochs + 1):
        train_running_loss = []
        model.train()
        start = time.time()
        for batch, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)

            if not finetune:
                # Compute L1 loss component
                loss = loss + args.weight_decay * sum(torch.abs(p).sum() for p in model.parameters())
            train_running_loss.append(loss.item())
            optimizer.zero_grad()   # clear gradients for next train
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            if mask is not None:
                maskgradient(model, mask, size_mask)
            # perform a single optimization step (parameter update)
            optimizer.step()
        end = time.time()
        t += (end-start)
        
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







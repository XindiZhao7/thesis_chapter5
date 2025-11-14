#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from icp import pValues, calculate_q


# In[ ]:
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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


def evaluate_ensemble(model, calib_loader, test_loader, eps, num_samples, classes, sample):
    correct = 0
    model.eval()
    
    corrects = np.zeros(num_samples, dtype=int)
    with torch.no_grad():
        for X_cal, y_cal in calib_loader:
            X_cal, y_cal = X_cal.to(device), y_cal.to(device)
            outputs = torch.zeros(num_samples, len(y_cal), classes).to(device)
            for i in range(num_samples):
                outputs[i] = model(X_cal, sample=sample)
            output = outputs.mean(0)
            cal_ncs = output
            calAlphas = cal_ncs[[torch.arange(0, len(y_cal)), y_cal]]
        
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = torch.zeros(num_samples, len(y_test), classes).to(device)
            for i in range(num_samples):
                outputs[i] = model(X_test, sample=sample)
            output = outputs.mean(0)
            test_ncs = output
            pvals = pValues(calAlphas, test_ncs, randomized=False)
            
            index = np.expand_dims(np.argwhere(pvals>eps), axis=2).tolist()
            prediction = torch.zeros_like(test_ncs)
            for i in index:
                prediction[i] = 1
            print('**************************************************************')
            mean_size = len(index)/len(y_test)
            print('Ineff: %f' %mean_size)

            correct_ = np.array([prediction[i, y_test[i]].cpu().numpy() for i in range(len(y_test))])
            n_correct = len(np.where(correct_==1)[0])
            coverage = n_correct/len(y_test)
            print('Err: %.1f' %(100*coverage))            
            
            y_hat = torch.argmax(test_ncs, axis=1)
            acc = torch.eq(y_hat, y_test).float().sum().item()/y_test.size(0)
            print('Acc: %f' % (acc*100))
            
    return mean_size, coverage, acc
    

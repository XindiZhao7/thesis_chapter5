#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import numpy as np

# calculate pValues with conformity scores
def pValues(calibrationAlphas,testAlphas,randomized=False):
    calibrationAlphas = calibrationAlphas.cpu().numpy()
    testAlphas = testAlphas.cpu().numpy()
    sortedCalAlphas = np.sort(calibrationAlphas)   
    rightPositions = np.searchsorted(sortedCalAlphas,testAlphas,side='right')

    if randomized:
        leftPositions = np.searchsorted(sortedCalAlphas,testAlphas)
        ties  = rightPositions-leftPositions+1   # ties in cal set plus the test alpha itself
        randomizedTies = ties * np.random.uniform(size=len(ties))
        return  (leftPositions + randomizedTies)/(len(calibrationAlphas)+1)
    else:
        return  (rightPositions + 1)/(len(calibrationAlphas)+1)


# In[ ]:


def calculate_q(alphas, epsilon):
    sorted_alphas, _ = torch.sort(alphas)
    n_cal = len(sorted_alphas)
    j = torch.floor(torch.Tensor([(n_cal+1)*epsilon])).long() - 1
    q = sorted_alphas[j]
    
    return q


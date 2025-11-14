#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.gate_layer import GateLayer
from layers.bayes_layer import BayesianLinear

# In[ ]:
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        
        self.fc1   = nn.Linear(input_dim, 500, bias=False)
        self.gate1 = GateLayer(500, 500, [1, -1])
        self.fc2   = nn.Linear(500, 300, bias=False)
        self.gate2 = GateLayer(300, 300, [1, -1])

        self.out   = nn.Linear(300, output_dim, bias=False)

    def forward(self, x):
        out = x.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.gate1(out)
        out = F.relu(self.fc2(out))
        out = self.gate2(out)
        out = self.out(out)
        
        return out


# In[ ]:


class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        
        self.fc1   = nn.Linear(input_dim, 1024, bias=False)
        self.gate1 = GateLayer(1024, 1024,[1, -1])
        self.fc2   = nn.Linear(1024, 1024, bias=False)
        self.gate2 = GateLayer(1024, 1024,[1, -1])
        self.fc3   = nn.Linear(1024, 512, bias=False)
        self.gate3 = GateLayer(512, 512,[1, -1])
        self.out   = nn.Linear(512, output_dim, bias=False)

    def forward(self, x):
        out = x.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.gate1(out)
        out = F.relu(self.fc2(out))
        out = self.gate2(out)
        out = F.relu(self.fc3(out))
        out = self.gate3(out)
        out = self.out(out)
        return out

class BMLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BMLP1, self).__init__()
        self.fc1 = BayesianLinear(input_dim, 500)
        self.gate1 = GateLayer(500,500,[1, -1])
        self.fc2 = BayesianLinear(500, 300)
        self.gate2 = GateLayer(300,300,[1, -1])
        self.out = BayesianLinear(300, output_dim)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x, sample))
        x = self.gate1(x)
        x = F.relu(self.fc2(x, sample))
        x = self.gate2(x)
        x = F.log_softmax(self.out(x, sample), dim=1)
        return x
    
    def log_prior(self):
        return self.fc1.log_prior \
               + self.fc2.log_prior \
               + self.out.log_prior
    
    def log_variational_posterior(self):
        return self.fc1.log_variational_posterior \
               + self.fc2.log_variational_posterior \
               + self.out.log_variational_posterior
    
    def sample_elbo(self, input, target, classes, batch_size, num_batches, samples=10):
        outputs = torch.zeros(samples, batch_size, classes).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_variational_posteriors = torch.zeros(samples).to(device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

    

class BMLP2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BMLP2, self).__init__()
        self.fc1 = BayesianLinear(input_dim, 1024)
        self.gate1 = GateLayer(1024,1024,[1, -1])
        self.fc2 = BayesianLinear(1024, 1024)
        self.gate2 = GateLayer(1024,1024,[1, -1])
        self.fc3 = BayesianLinear(1024, 512)
        self.gate3 = GateLayer(512,512,[1, -1])
        self.out = BayesianLinear(512, output_dim)
    
    def forward(self, x, sample=False):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x, sample))
        x = self.gate1(x)
        x = F.relu(self.fc2(x, sample))
        x = self.gate2(x)
        x = F.relu(self.fc3(x, sample))
        x = self.gate3(x)        
        x = F.log_softmax(self.out(x, sample), dim=1)
        return x
    
    def log_prior(self):
        return self.fc1.log_prior \
               + self.fc2.log_prior \
               + self.fc3.log_prior \
               + self.out.log_prior    
    
    def log_variational_posterior(self):
        return self.fc1.log_variational_posterior \
               + self.fc2.log_variational_posterior \
               + self.fc3.log_variational_posterior \
               + self.out.log_variational_posterior

    def sample_elbo(self, input, target, classes, batch_size, num_batches, samples=10):
        outputs = torch.zeros(samples, batch_size, classes).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_variational_posteriors = torch.zeros(samples).to(device)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood
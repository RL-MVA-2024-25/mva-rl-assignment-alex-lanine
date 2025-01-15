#standard imports
import numpy as np 
import torch; import torch.nn as nn 
from gymnasium.wrappers import TimeLimit
import itertools 
import random 

#local imports
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


#replay buffer - based on inclass example 
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.device = device #add gpu capability 
        self.capacity = capacity 
        self.data = []
        self.index = 0 
        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size):
        #sample random data
        batch = random.sample(self.data, batch_size) 
        #unzip batch
        unzipped_batch = list(zip(*batch))
        #convert to numpy 
        numpy_arrays = [np.array(field) for field in unzipped_batch]
        #covert to tensor
        tensors = [torch.Tensor(array) for array in numpy_arrays]
        #move to device
        tensors_on_device = [tensor.to(self.device) for tensor in tensors]
        return tensors_on_device 
    
    def __len__(self):
        return len(self.data)
import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from multiprocessing import  Pool
import time
from scipy.spatial import distance

class P2PDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.size(0)

    def add_data(self, feature, label):
        self.feature = torch.cat((self.feature, feature), 0)
        self.label = torch.cat((self.label, label), 0)


class P2PEnvironment(object):
    def __init__(self, config, seed):
        self.config = config
        self.n = config['points_per_iter']
        self.m = config['feature_dim']
        self.d = config['label_dim']
        self.eps = config['epsilon']
        self.eta = config['eta']
        self.KA = config['KA_norm']
        
        self.r_mean = config['mean_volunteer_attributes']
        self.r_std = config['std_volunteer_attributes']
        self.n_s = config['num_sources']
        self.n_d = config['num_destinations']
        
        self.avg_data_capacity = config['avg_data_capacity']
        self.std_data_capacity = config['std_data_capacity']
        self.x_tilde_attributes = config['x_tilde_attributes']
        self.r_attributes = config['r_attributes']

        
        np.random.seed(seed)
        self.A = (2 * np.random.rand(self.d, self.m) - 1)
        self.A = self.A/np.linalg.norm(self.A, 1) * np.random.rand() * self.KA
        self.mu = 2 * np.random.rand(self.d) - 1
        self.mu = self.mu/np.linalg.norm(self.mu, 2) * np.random.rand()        
        self.num_sources = config['num_sources']
        self.num_destinations = config['num_destinations']
        
                
        self.setup_locations()
        self.generate_volunteer_info()
            
        self.get_new_data(0)
        self.dataset = P2PDataset(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))

    def setup_locations(self):
        self.source_locations = np.random.random((self.num_sources,2))
        self.destination_locations = np.random.random((self.num_destinations,2))
        self.source_destination_distances = distance.cdist(self.source_locations,self.destination_locations)
        
    def generate_volunteer_info(self):
        self.volunteer_capacities = np.random.normal(self.config['avg_capacity'],self.config['std_capacity'],(self.d))
        self.volunteer_range = np.random.normal(self.config['avg_range'],self.config['std_range'],(self.d))
        self.r = np.array([self.volunteer_capacities,self.volunteer_range]).T
        
    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)

    def generate_label(self):
        self._y = self.A.dot(self._x[:,:-self.x_tilde_attributes].T).T + np.random.normal(0, self.eps, self.d)
        self._y[self._y>0.5] = 1
        self._y[self._y<0.5] = 0

    def generate_data(self):
        self._x = 2 * np.random.rand(self.n, self.m) - 1
        self._x = self._x / (np.tile(np.linalg.norm(self._x, 2, 1)/np.random.rand(self.n), (self.m, 1)).T)

        self.data_sources = np.random.randint(0,self.num_sources,(self.n))
        self.data_destinations = np.random.randint(0,self.num_destinations,(self.n))
        self.data_distances = []
        
        for i in range(self.n):
            source = self.data_sources[i]
            destination = self.data_destinations[i]
            self.data_distances.append(self.source_destination_distances[source][destination])
    
        self.data_distances = np.array(self.data_distances)
        self.data_capacities = np.random.normal(self.avg_data_capacity,self.std_data_capacity,(self.n))
        
        for column in [self.data_sources,self.data_destinations,self.data_distances,self.data_capacities]:
            self._x = np.append(self._x, np.reshape(column,(len(column),1)), axis=1)
            
    def mask_function(self,trip_number,volunteer_number):
        volunteer_capacity = self.r[volunteer_number][0]
        volunteer_range = self.r[volunteer_number][1]

        # Check if the capacity <= capacity, and the range <= range
        capacity_valid = volunteer_capacity >= self._x[trip_number][self.m+3]
        range_valid = volunteer_range >= self._x[trip_number][self.m+2]
        
        if range_valid and capacity_valid:
            return 1
        else:
            return 0
            
    def generate_mask(self):
        # From self._x and self.volunter_info, generate the nxd 0-1 matrix, mask
        self._m = np.zeros((self.n,self.d))
        
        for i in range(self.n):
            for j in range(self.d):
                self._m[i][j] = self.mask_function(i,j)
        
    def get_new_data(self, epoch_id):
        self.generate_data()
        self.generate_mask()
        self.generate_label()
        
        return self._x, self._y

    def add_to_data_loader(self):
        self.dataset.add_data(feature=torch.Tensor(self._x), label=torch.Tensor(self._y))

    def get_reward(self, action):
        ro = np.sum(self._y * action, 1)
        rb = action.dot(self.mu) + np.random.normal(0, self.eta, self.n)
        return ro, rb
    
    def get_match_reward(self, matches):
        return len(matches)

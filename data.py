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
    def __init__(self, feature, label, mask):
        self.feature = feature
        self.label = label
        self.mask = mask

    def __getitem__(self, index):
        return self.feature[index], self.label[index], self.mask[index]

    def __len__(self):
        return self.feature.size(0)

    def add_data(self, feature, label, mask):
        self.feature = torch.cat((self.feature, feature), 0)
        self.label = torch.cat((self.label, label), 0)
        self.mask = torch.cat((self.mask,mask),0)


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
        self.x_tilde_mean = config['mean_task_attributes']
        self.x_tilde_std = config['std_task_attributes']
        
        self.r_attributes = len(self.r_mean)
        self.x_tilde_attributes = len(self.x_tilde_mean)

        np.random.seed(seed)
           
        self.generate_underlying_parameters()
        self.generate_random_locations()
        self.generate_volunteer_info()
            
        self.get_new_data(0)
        self.dataset = P2PDataset(feature=torch.Tensor(self._x), label=torch.Tensor(self._y), mask=torch.Tensor(self._m))

    def generate_underlying_parameters(self):
        """Genreate the A and mu variables, which are used to generate
            the labels (c = Ax) and represent the unknown part of the loss
            
            Arguments: Nothing
            
            Returns: Nothing
            
            Side Effects: Sets A and mu attributes
        """
        self.A = (2 * np.random.rand(self.d, self.m) - 1)
        self.A = self.A/np.linalg.norm(self.A, 1) * np.random.rand() * self.KA
        self.mu = 2 * np.random.rand(self.d) - 1
        self.mu = self.mu/np.linalg.norm(self.mu, 2) * np.random.rand()     
        
    def generate_random_locations(self):
        """Generate a list of source locations (which refer to food exporters, such as grocery stores)
            and destination locations (which refer to food importers, such as food banks)
            
            Arguments: Nothing
            
            Return: Nothing
            
            Side Effects: Sets source_locations and destination_locations to be n_s x 2 and n_d x 2 matrices
        """
        
        self.source_locations = np.random.random((self.n_s,2))
        self.destination_locations = np.random.random((self.n_d,2))
        self.source_destination_distances = distance.cdist(self.source_locations,self.destination_locations)
        
    def generate_volunteer_info(self):
        """Generate the volunteer attributes for heterogenous agents, and store this in the r variable
            Each attribute is generated through a normal distribution, where the mean and std are 
            defined in the config
            
            Arguments: Nothing
            
            Returns: Nothing
            
            Side Effects: Sets the r variable representing volunteer attributes
        """
        
        volunter_attributes = []
        for i in range(self.r_attributes):
            attribute = np.random.normal(self.r_mean[i],self.r_std[i],(self.d))
            volunter_attributes.append(attribute)
        
        self.r = np.array(volunter_attributes).T
        
    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True)

    def generate_label(self):
        """Generate a random 0-1 label based on input x and matrix A
            y = A*x, for the non-extra attributes, with some Gaussian noise 
        
        Arguments: Nothing
        
        Returns: Nothing
        
        Side Effects: Sets the self._y variable 
        """
        
        self._y = self.A.dot(self._x[:,:-(self.x_tilde_attributes+3)].T).T + np.random.normal(0, self.eps, self.d)
        self._y[self._y>0.5] = 1
        self._y[self._y<0.5] = 0

    def generate_data(self):
        """Generate a random data point x, along with supplemental information x_tilde
            x_tilde captures source-destination-distance information, in addition to any 
            other features necesary 
            
            Arguments: Nothing
            
            Returns: Nothing
            
            Side Effects: Sets the self._x vector
        """
        
        self._x = 2 * np.random.rand(self.n, self.m) - 1
        self._x = self._x / (np.tile(np.linalg.norm(self._x, 2, 1)/np.random.rand(self.n), (self.m, 1)).T)

        self.data_sources = np.random.randint(0,self.n_s,(self.n))
        self.data_destinations = np.random.randint(0,self.n_d,(self.n))
        self.data_distances = [self.source_destination_distances[source][destination] for source, destination \
                               in zip(self.data_sources, self.data_destinations)]
        self.data_distances = np.array(self.data_distances)
        distance_attributes = [self.data_sources,self.data_destinations,self.data_distances]
        
        x_tilde_attributes = []
        for i in range(self.x_tilde_attributes):
            x_tilde_attributes.append(np.random.normal(self.x_tilde_mean[i],self.x_tilde_std[i],(self.n)))
                
        for column in distance_attributes + x_tilde_attributes:
            self._x = np.append(self._x, np.reshape(column,(len(column),1)), axis=1)
            
    def mask_function(self,trip_number,volunteer_number):
        """Create a mask function, which says which trip-volunteer combinations are valid, 
            based on external factors (x_tilde)
            
        Arguments: 
            trip_number: Which trip this is; 0<=trip_number<self.n
            volunter_number: Which volunteer we're trying to match; 0<=volunteer_number<self.d
    
        Returns: 0 or 1, depending on if a trip is valid
        """
        
        volunteer_capacity = self.r[volunteer_number][0]
        volunteer_range = self.r[volunteer_number][1]

        capacity_valid = volunteer_capacity >= self._x[trip_number][self.m+3]
        range_valid = volunteer_range >= self._x[trip_number][self.m+2]
        
        return int(capacity_valid and range_valid)

    def generate_mask(self):
        """Call the mask function for every combination of trip and volunteer
        
        Arguments: Nothing
        
        Returns: Nothing
        
        Side Effects: Sets the mask, self._m
        """
        
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
        self.dataset.add_data(feature=torch.Tensor(self._x), label=torch.Tensor(self._y), mask=torch.Tensor(self._m))

    def get_reward(self, action):
        """Get the reward by seeing which actions are valid (through self._y and self._m)
        And also the hidden error (through self._mu)
        
        Arguments: 
            Action: n x d numpy matrix
           
        Returns:
            Reward from optimizing and reward from bandit
        """
        
        ro = np.sum(-self._y * action * self._m, 1)
        rb = (action*self._m).dot(self.mu) + np.random.normal(0, self.eta, self.n)
        return ro, rb
    
    def get_match_reward(self, matches):
        return len(matches)

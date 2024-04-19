import numpy as np
import shutil, time, os, requests, random, copy
import time
import glob
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import torchvision
import matplotlib.pyplot as plt
from collections import defaultdict

class Kids450(Dataset):
    def __init__(self,phase,file_paths,s = 0.5):
        self.phase = phase #string for "train" or "valid" to separate the phase
        self.variable = "kaks"
        if self.phase == "train":
            self.sample_range = [0,8000]
        if self.phase == "val":
            self.sample_range = [8000,10000]
        self.file_paths = file_paths #file path for
        self.sample_indices, self.x2_idx = self.generate_sample_indices() #sample index pairs (file_nr, sample_nr)
        self.length = len(self.sample_indices)
     
        ###Adding transforms      
        #https://github.com/pytorch/vision/issues/566 might become slow
        self.transforms = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),#random flip same as 180 degrees
                                              transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.5),
                                              transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                              #AddGaussianNoise(mean=0., std=1.)
                                            ])                
    def generate_sample_indices(self):
        """
        This function gives me basically a lookup table where I will have s
        ample file and sample index pairs scrambled for randomizing effect
        
        """
        sample_indices = []        
        for i,file_path in enumerate(self.file_paths):
            #with h5py.File(file_path, 'r') as f:
                #num_samples = len(f["kappa"]) #no need since I know the amount of files
                #sample_indices.extend([(i,j) for j in range(num_samples)])
            
            sample_indices.extend([(i,j) for j in range(self.sample_range[0],self.sample_range[1])])
                
        x2_idx = [tup[1] for tup in sample_indices] #list of sample_nr randomized from the same file for the augmentation of the image
        np.random.shuffle(x2_idx)
        np.random.shuffle(sample_indices) #This shuffles the indices in place
        return sample_indices, x2_idx
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.length

    
    #SO HERE WE See that the getitem gives you two images, but the images returned are augmentations and preprocess of one instanse
    #THe self.augment gives one random augmentation
    def __getitem__(self,idx):
        
        x1_idx =  self.sample_indices[idx]
        x2_idx =  self.x2_idx[idx]
        """
        Doing an workaround with the custom collate function.  
        So __getitem__ will pass indices to collate function in order not to touch the correct order of indeces when shuffeling
        """   
        #print(idx)
        return x1_idx, x2_idx
    

    
    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        """
        Should implement this later for scrambling the dataset at each epoch
        """
        self.imgarr = self.imgarr[random.sample(population = list(range(self.__len__())),k = self.__len__())]
    
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame, transformations = None):
        
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        
        return frame
    
##Gausian noise
#https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
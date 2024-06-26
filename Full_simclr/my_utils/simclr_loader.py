import numpy as np
import shutil, time, os, requests, random, copy
import time
import glob
import os
import pickle as pkl
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
    def __init__(self,phase,file_paths,resolution):
        self.phase = phase #string for "train" or "valid" to separate the phase
        if self.phase == "train":
            self.sample_range = [0,8000]
        if self.phase == "val":
            self.sample_range = [8000,10000]
            
        self.file_paths = file_paths #file path for
        self.sample_indices, self.x2_idx = self.generate_sample_indices() #sample index pairs (file_nr, sample_nr)
        #self.MEAN, self.STD = self.get_data_stats()
        self.length = len(self.sample_indices)
        self.resolution = resolution
     
        ###Adding transforms      
        #https://github.com/pytorch/vision/issues/566 might become slow
        #self.transforms = Kids450_augmentations(self.resolution)
        self.transforms =  transforms.Compose([transforms.RandomVerticalFlip(p=0.5),#random flip same as 180 degrees
                                    transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.5),
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.RandomResizedCrop(resolution,(0.8,1.0)),
                                    #transforms.RandomCrop(resolution),
                                    #AddGaussianNoise(mean=0., std=1.)
                                    ]) 
        self.transforms_valid = transforms.Compose([transforms.RandomCrop(resolution)])
        #self.data_stats = self.get_data_stats()
    def generate_sample_indices(self):
        """
        This function gives me basically a lookup table where I will have s
        ample file and sample index pairs scrambled for randomizing effect
        
        """
        sample_indices = []        
        for i,_ in enumerate(self.file_paths):
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
        return x1_idx, x2_idx,self
        #return x1_idx, x2_idx,self.resolution,self.file_paths,self.augment,self
    
    #make a function that checks if first 3 samples indeed are shuffled
    def check_shuffling(self):
        """
        This function checks if the first 3 samples are shuffled
        """
        return self.sample_indices[:3]

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        """
        Should implement this later for scrambling the dataset at each epoch
        """
        random.shuffle(self.sample_indices)
    
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame):
        """
        Will not use it anyomore actually. Will create a custom collate function instead
        """
        
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            if self.resolution != 128:
                frame = self.transforms_valid(frame)
            return frame
        
        return frame
    
    def preprocess(self, frame):
        """
        Preprocesses the frame
        """
        #print("Preprocessing",frame.shape, flush=True)
        frame = (frame-self.MEAN)/self.STD
        return frame
    
    def get_data_stats(self):
        """
        This function is used to get the mean and std of the dataset
        """
        filepath ="/cluster/work/refregier/atepper/kids_450/full_data/kids450_train_stats.pkl"
        with open(filepath, 'rb') as f:
            data_stats = pkl.load(f)
            MEAN = data_stats["Overall_mean"]
            STD = data_stats["Overall_std"]
        return MEAN, STD  
    
#def Kids450_augmentations(resolution):
#    """
#    This function returns a set of augmentations for the Kids450 dataset
#    """
#
#    augmentations = transforms.Compose([transforms.RandomVerticalFlip(p=0.5),#random flip same as 180 degrees
#                                    transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.5),
#                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
#                                    transforms.RandomCrop(resolution),
#                                    #AddGaussianNoise(mean=0., std=1.)
#                                    ]) 
#
#    return augmentations
#




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
    
def kids450_files_localscratch():
    """
    This function is used to get the file paths for the Kids450 dataset
    when running on the local scratch.
    return: file_paths_train, file_paths_test

    """

    tmp_dir = os.environ.get('TMPDIR') # Get the list of input files at local scratch
    input_directory = tmp_dir + "/kids_450_h5_files/" + "kids_450_h5"
    file_paths_train = glob.glob(os.path.join(input_directory, '*train.h5'))
    file_paths_test = glob.glob(os.path.join(input_directory, '*test.h5'))

    return file_paths_train, file_paths_test




def kids450_files_cluster():
    ### for testing on euler  jupyter notebook
    """
    This function is used to get the file paths for the Kids450 dataset
    when running on the jupyterhub.
    return: file_paths_train, file_paths_test 
    """
    input_directory = "/cluster/work/refregier/atepper/kids_450/full_data/kids_450_h5"
    #file_pattern = '*train.h5'
    file_paths_train = glob.glob(os.path.join(input_directory, '*train.h5'))
    file_paths_test = glob.glob(os.path.join(input_directory, '*test.h5'))

    return file_paths_train, file_paths_test
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

class mae_kids450(Dataset):
    def __init__(self,phase,file_paths,resolution):
        self.phase = phase #string for "train" or "valid" to separate the phase
        self.file_paths = file_paths #file path for
        if self.phase == "train":
            self.sample_range = [0,8000]
        if self.phase == "val":
            self.sample_range = [8000,10000]
        if self.phase == "test":
            # The test file will have separate files so will start from 0 to 2000
            self.sample_range = [0,2000]

            
        self.sample_indices = self.generate_sample_indices() #sample index pairs (file_nr, sample_nr)
        self.length = len(self.sample_indices)
        self.resolution = resolution
        self.data_stats_dict = data_stats_dict()
     
        ###Adding transforms      
        #https://github.com/pytorch/vision/issues/566 might become slow  Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        #self.transforms = Kids450_augmentations(self.resolution)
        self.transforms =  transforms.Compose([transforms.RandomResizedCrop(size=(self.resolution,self.resolution), scale=(0.7, 1.0), interpolation=3),  # 3 is bicubic
                                    #transforms.RandomVerticalFlip(p=0.5),#random flip same as 180 degrees
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply([transforms.RandomRotation((270, 270))], p=0.5),
                                    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                                    transforms.Normalize(( 1.7944e-13,-9.2091e-14,-1.8305e-13, 4.7488e-13),(0.0079,0.0065,0.0093,0.0116)),
                                    #transforms.RandomResizedCrop(size=(self.resolution,self.resolution), scale=(0.7, 1.0)),
                                    #transforms.RandomCrop(resolution),
                                    #AddGaussianNoise(mean=0., std=1.)
                                    ]) 
        
        self.transforms_valid = transforms.Compose([transforms.RandomResizedCrop(size=(self.resolution,self.resolution), interpolation=3),  # 3 is bicubic
                                    transforms.Normalize(( 1.7944e-13,-9.2091e-14,-1.8305e-13, 4.7488e-13),(0.0079,0.0065,0.0093,0.0116)),
                                    ]) 
        
    def generate_sample_indices(self):
        """
        This function gives me basically a lookup table where I will have 
        sample file and sample index pairs scrambled for randomizing effect
        
        """
        sample_indices = []        
        for i,_ in enumerate(self.file_paths):
            sample_indices.extend([(i,j) for j in range(self.sample_range[0],self.sample_range[1])])
        #since the test set does not need to be shuffled i create a not conditional statement
        if self.phase != "test":
            np.random.shuffle(sample_indices)#shuffle indeces in place    
        return sample_indices
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.length


    def __getitem__(self,idx):
        """
        This function takes index  pair of 57 fails and 1 of 10000 elements and passesi it
        to the cosmo_collate function to load the data there.
        """
        
        x1_idx =  self.sample_indices[idx]
        return x1_idx,self
    

    
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
            #if self.resolution != 128:
            #    frame = self.transforms_valid(frame)
            #return frame
            frame = self.transforms_valid(frame)
            return frame

        
        return frame
    
    def standardize_label(self,label_batch):
        """
        This function will standardize the frame
        """
        mean_labels = self.data_stats_dict["mean_labels"] 
        std_labels = self.data_stats_dict["label_std_deviation"]
        label_batch = (label_batch - mean_labels) / std_labels.unsqueeze(0)
        return label_batch
    


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

def data_stats_dict():
    """
    This function will give me a dictory with the mean and std of the dataset for labels
    and images. But I will use only the labels since the images are already centered
    """
    pickle_file_path = "/cluster/work/refregier/atepper/kids_450/full_data/kids450_test_stats.pkl"
    # Open the pickle file and load the data
    with open(pickle_file_path, 'rb') as f:
        data_dict = pkl.load(f)
    return data_dict
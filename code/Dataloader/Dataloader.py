import numpy as np
import pandas as pd
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
import pickle

#set seed for reproductibility
def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)

    
class Kids450(Dataset):
    def __init__(self,phase,file_paths,s = 0.5):
        self.phase = phase #string for "train" or "valid" to separate the phase
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
        
        """
        Now thi
        """
        #I have to take an image from the same cosmologie file for positive example
        # I have to make sure that randomly the same element is taken
        
        file_idx, sample_idx = self.sample_indices[idx]
        with h5py.File(self.file_paths[file_idx], "r") as f:
            data = f["kappa"]
            x1 = data[sample_idx]
            #x2 = data[sample_idx] same 
            x2 = data[self.x2_idx[sample_idx]]# "augmentation from same file"

        # why it divides by 255.0, beacuse original images in 0-255 range of pixel values then you get it to [0,1]
        x1 = x1.astype(np.float32)/255.0# why it divides by 255.0, beacuse original images in 0-255 range of pixel values then you get it to [0,1]
        x2 = x2.astype(np.float32)/255.0
        
        x1 = self.augment(torch.from_numpy(x1))
        x2 = self.augment(torch.from_numpy(x2))
        
        return x1, x2
    
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
    
    
    
    
# Get the list of input files
input_directory = "/cluster/work/refregier/atepper/kids_450_h5"
file_pattern = '*train.h5'
file_paths = glob.glob(os.path.join(input_directory, file_pattern))
#initialize the dataset
number_of_workers = [10,12,15,17,20]
#number_of_workers = [2,4,5,8]
#number_of_workers = [15,20,25]
#number_of_workers = [20,30,40]
phase = "train" #Dataloader phase
batch_cap = 10   #how many batches to go trough in one round
batch_size = 128 #size of batch
time_per_workers = []

for num_workers in number_of_workers:
    print(num_workers)
    dg = Kids450(phase,file_paths)
    dl = DataLoader(dg,batch_size = batch_size , drop_last=True,num_workers = num_workers)
    #test for 10 batches of size 128
    time_list = []
    for Round in range(5):
        start_time = time.time()
        count = 0
        #for i,(x1,x2) in enumerate(dl):
        for i,(x1,x2) in enumerate(dl):
            #print(i)
            #print(i)
            count += 1
            #print(i)
            if i == batch_cap:
                break
        end_time = time.time()
        #print(count)
        #print(count)#okay it still has 312 batches which is okay
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
    time_per_workers.append(np.mean(time_list))

data_dict = {
            "time_per_workers" : time_per_workers,
            "batchcap" : batch_cap,
            "num_cores" : number_of_workers,
            "batch_size" : batch_size
            }
    
save_name = 'timing_test_train_phase_py_batch.pkl'
save_path = "/cluster/home/atepper/master_thesis/master_thesis/data/test_data/"
with open(save_path+save_name, 'wb') as f:
    pickle.dump(data_dict, f)
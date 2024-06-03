import torch
from collections import defaultdict
import h5py
from my_utils.simclr_loader import  kids450_files_localscratch

def collate_fn(idxs):
        """
        Input batch_indices are a batch of indices of required batch size.
        Now I will load the files more efficiently.
        
        Return x1 and x2 where x1 is the original and x2 the augmented version. 
        """      
        batch_size = len(idxs)
        
        x1_indices = [item[0] for item in idxs] #list of  tuples (file_idx, element_idx_within_file)
        x2_indices = [item[1] for item in idxs] #element number from randomized lookup table to use for taking image from same cosmology/file
        dataset_obj = idxs[0][2]#get dataset object from first element
        resolution = dataset_obj.resolution#resolution from dataset object
        file_paths = dataset_obj.file_paths#file paths from dataset object
        augment = dataset_obj.augment
        #preprocess = dataset_obj.preprocess

        #group the element indeces belonging to a certain file in order to
        file_groups = defaultdict(list)
        for file_index, element_index in x1_indices:
            file_groups[file_index].append(element_index)
        # Convert defaultdict to regular dictionary and sort by file index
        #sort by file and elements
        sorted_file_groups = sorted(file_groups.items())
        
        #create empty tensors for batches
        x1_batch = torch.empty((batch_size,1 ,4,resolution,resolution)) #the 1 is added as tensor.unsqueeze(1)
        x2_batch = torch.empty((batch_size,1 ,4,resolution,resolution)) #the 1 is added as tensor.unsqueeze(1)
        #define augmentations and file paths
        #augment = Kids450_augmentations(resolution)
        #file_paths = kids450_files_localscratch()

        count = 0#count for adding elements to batch

        for file_idx, elements in sorted_file_groups:  #extract file idx and corrseponding elements
            with h5py.File(file_paths[file_idx], "r") as f:
                data = f["kappa"]
                for elem in elements:

                    x1 = data[elem] #get element from file
                    x2 = data[x2_indices[count]] #get different element from same file
                    
                    
                    #x1 = x1/255.0# why it divides by 255.0, beacuse original images in 0-255 range of pixel values then you get it to [0,1]
                    #x2 = x2/255.0
                    
                    #augment
                    x1 = augment(torch.from_numpy(x1))
                    x2 = augment(torch.from_numpy(x2))
                    
                    #preprocess
                    #x1 = preprocess(x1)
                    #x2 = preprocess(x2)


                    x1 = x1.unsqueeze(0) #to add 1 dimesion to fit the original batch shape
                    x2 = x2.unsqueeze(0) #to add 1 dimesion to fit the original batch shape
                    
                    
                    #add to batch
                    x1_batch[count] = x1
                    x2_batch[count] = x2
                    
                    #add to counter
                    count +=1
        return x1_batch,x2_batch




def collate__fn_valid(idxs):
        """
        Input batch_indices are a batch of indices of required batch size.
        Now I will load the files more efficiently.
        
        Return x1 and x2 where x1 is the original and x2 the augmented version. 
        """      
        batch_size = len(idxs)
        
        x1_indices = [item[0] for item in idxs] #list of  tuples (file_idx, element_idx_within_file)
        x2_indices = [item[1] for item in idxs] #element number from randomized lookup table to use for taking image from same cosmology/file
        dataset_obj = idxs[0][2]#get dataset object from first element
        resolution = dataset_obj.resolution#resolution from dataset object
        file_paths = dataset_obj.file_paths#file paths from dataset object
        preprocess = dataset_obj.preprocess

        #group the element indeces belonging to a certain file in order to
        file_groups = defaultdict(list)
        for file_index, element_index in x1_indices:
            file_groups[file_index].append(element_index)
        # Convert defaultdict to regular dictionary and sort by file index
        #sort by file and elements
        sorted_file_groups = sorted(file_groups.items())
        
        #create empty tensors for batches
        x1_batch = torch.empty((batch_size,1 ,4,resolution,resolution)) #the 1 is added as tensor.unsqueeze(1)
        x2_batch = torch.empty((batch_size,1 ,4,resolution,resolution)) #the 1 is added as tensor.unsqueeze(1)
        #file paths
        file_paths = kids450_files_localscratch()

        count = 0#count for adding elements to batch

        for file_idx, elements in sorted_file_groups:  #extract file idx and corrseponding elements
            with h5py.File(file_paths[file_idx], "r") as f:
                data = f["kappa"]
                for elem in elements:

                    x1 = data[elem] #get element from file
                    x2 = data[x2_indices[count]] #get different element from same file
                    
                    #normalize 
                    #x1 = x1/255.0# why it divides by 255.0, beacuse original images in 0-255 range of pixel values then you get it to [0,1]
                    #x2 = x2/255.0
                    
                    #no augmentations, just convert to tensor
                    x1 = torch.from_numpy(x1).unsqueeze(0)
                    x2 = torch.from_numpy(x2).unsqueeze(0)
                                 
                    #preprocess
                    #x1 = preprocess(x1)
                    #x2 = preprocess(x2)
                    
                    #add to batch
                    x1_batch[count] = x1
                    x2_batch[count] = x2
                    
                    #add to counter
                    count +=1
        return x1_batch,x2_batch




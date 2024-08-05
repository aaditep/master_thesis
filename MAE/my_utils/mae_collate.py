import torch
from collections import defaultdict
import h5py
from my_utils.mae_dataloader import  kids450_files_localscratch

def mae_collate_fn(idxs):
        """
        Input batch_indices are a batch of indices of required batch size.
        Now I will load the files more efficiently.
        
        Return x1 and x2 where x1 is the original and x2 the augmented version. 
        """      
        batch_size = len(idxs)

        x1_indices = [item[0] for item in idxs] #list of  tuples (file_idx, element_idx_within_file)
        dataset_obj = idxs[0][1]#get dataset object from first element
        resolution = dataset_obj.resolution#resolution from dataset object
        file_paths = dataset_obj.file_paths#file paths from dataset object
        augment = dataset_obj.augment
        standardize_label = dataset_obj.standardize_label
        phase = dataset_obj.phase
        

        #group the element indeces belonging to a certain file in order to
        file_groups = defaultdict(list)
        for file_index, element_index in x1_indices:
            file_groups[file_index].append(element_index)
        # Convert defaultdict to regular dictionary and sort by file index
        #sort by file and elements
        sorted_file_groups = sorted(file_groups.items())
        
        #create empty tensors for batches
        x_batch = torch.empty((batch_size,4,resolution,resolution)) #empty tensor for images
        y_batch = torch.empty((batch_size,2)) #empty tensor for corresponding labels
        
        count = 0#count for adding elements to batch

        for file_idx, elements in sorted_file_groups:  #extract file idx and corrseponding elements
            #print("collecting")
            with h5py.File(file_paths[file_idx], "r") as f:
                x = f["kappa"]
                y = f["labels"]
                for elem in elements:

                    x1 = x[elem] #get the image from the file
                    y1 = torch.tensor(y[elem]) #get the label from the file
                    
                    
                    #augment
                    x1 = augment(torch.from_numpy(x1))#augment the image
                    x1 = x1 #to add 1 dimesion to fit the original batch shape

                    
                    #add to batch
                    x_batch[count] = x1
                    y_batch[count] = y1
                    
                    #add to counter
                    count +=1

        #preprocess the y_batch if not test set
        if phase != "test":
            y_batch = standardize_label(y_batch)

        return x_batch,y_batch#,sorted_file_groups
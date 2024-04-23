import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy, glob
import pickle
import h5py
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchsummary import summary
from my_utils.LARS import LARS
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from my_utils.Dataloader_class import Kids450
from my_utils.NT_xent_loss import SimCLR_Loss
import yaml

#fix "too many open files error"
import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')




def collate__faster_load(idxs):
        """
        Faster collate function for the dataloader. It loads the images and then augments them.
        The faster part is loading images belonging to the same file, not opening and closing the file for each image.

        Args:
            idxs: list of indices
        Returns:
            x1_batch: batch of augmented(if training phase) images
            x2_batch: batch of augmented(if training phase) images
        """
        
        batch_size = len(idxs)
        
        x1_indices = [item[0] for item in idxs] #list of  tuples (file_idx, element_idx_within_file)
        x2_indices = [item[1] for item in idxs] #element number from randomized lookup table to use for taking image from same cosmology/file
        
        #group the element indeces belonging to a certain file in order to
        file_groups = defaultdict(list)
        for file_index, element_index in x1_indices:
            file_groups[file_index].append(element_index)
        # Convert defaultdict to regular dictionary and sort by file index
        #sort by file and elements
        sorted_file_groups = sorted(file_groups.items())

       
        resolution = 128
        #create empty tensors for batches
        x1_batch = torch.empty((batch_size,1 ,4, resolution, resolution)) #the 1 is added as tensor.unsqueeze(1)
        x2_batch = torch.empty((batch_size,1 ,4, resolution, resolution)) #the 1 is added as tensor.unsqueeze(1)

        count = 0#count for adding elements to batch and take x2
        for file_idx, elements in sorted_file_groups:  #extract file idx and corrseponding elements
            with h5py.File(file_paths[file_idx], "r") as f:
                data = f["kappa"]
                for elem in elements:

                    x1 = data[elem] #get element from file
                    x2 = data[x2_indices[count]] #get different element from same file
                    
                    #normalize
                    x1 = x1/255.0# why it divides by 255.0, beacuse original images in 0-255 range of pixel values then you get it to [0,1]
                    x2 = x2/255.0
                    
                    #augment
                    x1 = dg.augment(torch.from_numpy(x1))# torch.from_numpy() changes into tensor in place wp copy
                    x2 = dg.augment(torch.from_numpy(x2)) 
                    
                    x1 = x1.unsqueeze(0) #to add 1 dimesion to fit the original batch shape
                    x2 = x2.unsqueeze(0) #to add 1 dimesion to fit the original batch shape
                    
                    
                    #add to batch
                    x1_batch[count] = x1
                    x2_batch[count] = x2
            
                    #add to counter
                    count +=1

                    #torch.from_numpy(x1)
        return x1_batch, x2_batch


#Set random seed for reproducibility


def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(seed = 16)

#The identity module gives the same output as the input
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

#The linearLayer module gives a single Linear layer with optional batch normalization layer
class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


#Set up the projection head with a linear or non-linear head
#This is the base encoder f(.)
class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x


class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL
        if self.base_model == 'resnet18':
            self.pretrained = models.resnet18(pretrained = True)
            print("loaded resnet 18", flush=True)
        elif self.base_model == 'resnet50':
            self.pretrained = models.resnet50(pretrained = True)
            print("loaded resnet 50", flush=True)
        #self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        

        if self.base_model == 'resnet18':
            self.projector = ProjectionHead(512, 512, 128)
        elif self.base_model == 'resnet50':
            self.projector = ProjectionHead(2048, 2048, 128)
    

    def forward(self,x):
        """Standard forward function for PyTorch models that 
        connects the  pretrined object with the projection head.
        So Here the model features are learned by the pretrained model and later can be used without
        the projection head for fine tuning or transfer learning in a downstream task.
        Args:
            x: input data
                            
        Returns: 
            xp: output of the projection head"""

        out = self.pretrained(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp
    
#Extra functions for model saving and loading and pltting features with TSNE

def save_model(model, optimizer, scheduler, current_epoch, name):
    """
    Save the model

    Args:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    scheduler : torch.optim.lr_scheduler : scheduler
    current_epoch : int : current epoch
    name : str : name of the model
    """
    out = os.path.join('./data/saved_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)

def plot_features(model, num_classes, num_feats, batch_size, current_epoch):
    """
    Plot the features with TSNE. THIS IS STILL A TODO-IMPLEMENTATION

    Args:
    model : PreModel : model
    num_classes : int : number of classes
    num_feats : int : number of features
    batch_size : int : batch size
    current_epoch : int : current epoch

    """
    preds = np.array([]).reshape((0,1))
    gt = np.array([]).reshape((0,1))
    feats = np.array([]).reshape((0,num_feats))
    model.eval()
    with torch.no_grad():
        for x1,x2 in valid_loader:
            x1 = x1.squeeze().to(device = 'cuda:0', dtype = torch.float)
            out = model(x1)
            out = out.cpu().data.numpy()#.reshape((1,-1))
            feats = np.append(feats,out,axis = 0)
    
    tsne = TSNE(n_components = 2, perplexity = 50)
    x_feats = tsne.fit_transform(feats)
    num_samples = int(batch_size*(valimages.shape[0]//batch_size))#(len(val_df)
    
    plt.clf()

    for i in range(num_classes):
        plt.scatter(x_feats[vallabels[:num_samples]==i,1],x_feats[vallabels[:num_samples]==i,0])
    plt.title(f"TSNE plot of features at epoch {current_epoch}")#plot title
    plt.legend([str(i) for i in range(num_classes)])#plot legend
    save_path = "./data/plots/" #for saving plot
    plt_name= f"TSNE_plot_epoch_{current_epoch}.pdf"#plot name
    plt.savefig(save_path+plt_name, format='pdf')#save plot
    #plt.show()

def plot_losses(tr_loss, val_loss,current_epoch):
    """
    Plot the training and validation loss

    Args:
    tr_loss : list : training loss
    val_loss : list : validation loss
    current_epoch : int : current epoch
    """
    plt.clf()
    plt.plot(tr_loss)
    plt.plot(val_loss)
    plt.title(f"Training and Validation Loss at epoch {current_epoch}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training Loss","Validation Loss"])
    save_path = "./data/plots/"
    plt_name= f"Loss_plot_epoch_{current_epoch}.pdf"
    plt.savefig(save_path+plt_name, format='pdf')
    #plt.show()
    
def load_config(config_file):
    """Load YAML configuration from file."""
    with open(config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def data_origin(origin : str, full_sample = True):
    """
    Get the data origin

    Args:
    origin : str : origin of the data
    full_sample : bool : use full sample or not (5% or 100% of data)
    """
    if origin == 'local_scratch':
        tmp_dir = os.environ.get('TMPDIR') # Get the list of input files at local scratch
        input_directory = tmp_dir + "/kids_450_h5"
        if full_sample == False:
            input_directory = tmp_dir + "/kids_450_h5_small_sample"
        file_pattern = '*train.h5'
        file_paths = glob.glob(os.path.join(input_directory, file_pattern))
        return file_paths

def prepare_datasets(file_paths : list):
    """
    Prepare the datasets for training and validation data

    Args:
    file_paths : list : list of file paths

    Returns:
    dg : Kids450 : dataset for training data
    vdg : Kids450 : dataset for validation data
    """
    dg = Kids450(phase = "train",file_paths = file_paths)
    vdg = Kids450(phase = "val",file_paths = file_paths)
    return dg, vdg


def prepare_dataloaders(dg, vdg, batch_size: int, num_workers: int):
    """
    Prepare the dataloaders for training and validation data

    Args:
    dg : Kids450 : dataset for training data
    vdg : Kids450 : dataset for validation data
    batch_size : int : batch size
    num_workers : int : number of workers for dataloader    

    Returns:
    train_loader : DataLoader : dataloader for training data
    valid_loader : DataLoader : dataloader for validation data   
    """
    train_loader = DataLoader(dg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = collate__faster_load,
                               pin_memory = True,
                               num_workers = num_workers)

    valid_loader = DataLoader(vdg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = collate__faster_load,
                               pin_memory = True,
                               num_workers = num_workers)

    return train_loader, valid_loader

class Trainer:
    """
    Trainer class to train the model
    """
    def __init__(
        self,
        model,
        train_data,
        optimizer,
        gpu_id,
        warmupscheduler,
        mainscheduler,
        criterion,
        save_every, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.warmupscheduler = warmupscheduler
        self.mainscheduler = mainscheduler
        self.loss = criterion
        self.save_every = save_every

    def _run_batch(self, x_i, x_j):
        """
        Run a single batch through the model and update the weights

        Args:
        x_i : torch.Tensor : input data
        x_j : torch.Tensor : input data
        """
        self.optimizer.zero_grad()
         # positive pair, with encoding
        z_i = self.model(x_i)
        z_j = self.model(x_j)
        loss = self.loss(z_i, z_j)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        """
        Run a single epoch through the model

        Args:
        epoch : int : epoch number
        """
        b_sz = len(next(iter(self.train_data))[0]) #get batch size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for (x_i,x_j) in self.train_data:
            x_i = x_i.squeeze().to(self.gpu_id).float()
            x_j = x_j.squeeze().to(self.gpu_id).float()
            self._run_batch(x_i, x_j)



    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "./data/saved_models_multi/checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)



def load_train_objs(batch_size):
    """
    Load the training objects

    Args:
    batch_size : int : batch size
    
    Returns:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    warmupscheduler : torch.optim.lr_scheduler.LambdaLR : warmup scheduler
    mainscheduler : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts : main scheduler
    criterion : SimCLR_Loss : loss function
    """

    model = PreModel('resnet18').to("cuda")
    print(f" Summary of the model: {summary(model, (4, 128, 128))}",flush=True)# Works now
    optimizer = LARS(
                    [params for params in model.parameters() if params.requires_grad],
                    lr=0.2,
                    weight_decay=1e-6,
                    exclude_from_weight_decay=["batch_normalization", "bias"],
                    )
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True) #decay the learning rate with the cosine decay schedule without restarts"
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)#scheduler for cosine decay
    criterion = SimCLR_Loss(batch_size = batch_size, temperature = 0.5)#loss function
    
    
    return model, optimizer, warmupscheduler, mainscheduler, criterion

def main(device, total_epochs,save_every, batch_size):
    """
    Main function to train the model

    Args:
    device : int : device id
    total_epochs : int : total epochs
    save_every : int : save every
    batch_size : int : batch size
    """
    model, optimizer, warmupscheduler, mainscheduler, criterion = load_train_objs(batch_size)
    trainer = Trainer(model, train_loader, optimizer, device, warmupscheduler, mainscheduler, criterion, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Simclr job')
    parser.add_argument('--config', type=str, default='./config_Simclr_kids450.yaml', help='config file path')
    args = parser.parse_args()
    ######With YAML CONFIG######################
    config_path = args.config
    config = load_config(config_path)
    file_paths = data_origin(config['origin'], config['full_sample'])
    dg, vdg = prepare_datasets(file_paths)
    train_loader, valid_loader = prepare_dataloaders(dg,vdg, config['batch_size'], config['num_workers'])


    device = 0  # shorthand for cuda:0
    main(device, config['total_epochs'],config['save_every'], config['batch_size'])

    
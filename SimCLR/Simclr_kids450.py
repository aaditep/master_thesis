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

#fix "too many open files error"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')





def collate__faster_load(idxs):
        """
        Input batch_indices are a batch of indices of required batch size.
        Now I will load the files more efficiently.
        
        Return x1 and x2 where x1 is the original and x2 the augmented version. 
        """
        #print(idxs)
        
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
        
        #create empty tensors for batches
        x1_batch = torch.empty((batch_size,1 ,4, 128, 128)) #the 1 is added as tensor.unsqueeze(1)
        x2_batch = torch.empty((batch_size,1 ,4, 128, 128)) #the 1 is added as tensor.unsqueeze(1)
        
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
        return x1_batch,x2_batch


#Set random seed for reproducibility


def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(seed = 16)



#Define the source of data on the local scratch of the node

#get the list of input from the local sctarch
tmp_dir = os.environ.get('TMPDIR')
# Get the list of input files
#small sample dataset
input_directory = tmp_dir + "/kids_450_h5_small_sample"
#full dataset
#input_directory = tmp_dir + "/kids_450_h5"
file_pattern = '*train.h5'
file_paths = glob.glob(os.path.join(input_directory, file_pattern))


batch_size = 128
num_workers = 39
#Initailize the dataloaders for training ("train") and validation ("val") data
dg = Kids450(phase = "train",file_paths = file_paths)
train_loader = DataLoader(dg,batch_size = batch_size , drop_last=True,prefetch_factor = 3, collate_fn = collate__faster_load, pin_memory = True, num_workers = num_workers)
#maybe add prefetch_factor = 3
vdg = Kids450(phase = "val",file_paths = file_paths)
valid_loader = DataLoader(vdg,batch_size = batch_size , drop_last=True, prefetch_factor = 3, collate_fn = collate__faster_load, pin_memory = True, num_workers = num_workers)
print("dataloaders initialized")



#Setup model
print("Settin up model",flush = True)

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
    
#Setup the Pretraining model
#It looks like the maxpool and the fc are just placeholders for the model to be able to load the weights

class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL
        if self.base_model == 'resnet18':
            elf.pretrained = models.resnet18(pretrained = True)
        elif self.base_model == 'resnet50':
            self.pretrained = models.resnet50(pretrained = True)
        #self.pretrained = models.resnet50(pretrained=True)
        
        #self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
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
    

    #Initialize the model
print("Initializing model, sending it to GPU and printing summary: ",flush = True)
model = PreModel('resnet50').to('cuda:0')
summary(model, (4, 128, 128))# Works now


#Declaration of optimizer, loss and scheduler


optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)
print("Initialized optimizer",flush = True)

# "decay the learning rate with the cosine decay schedule without restarts"
#SCHEDULER OR LINEAR EWARMUP
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

#SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#LOSS FUNCTION
criterion = SimCLR_Loss(batch_size = batch_size, temperature = 0.5)

print("Initialized Loss",flush = True)

#Extra functions for model saving and loading and pltting features with TSNE

def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join('./data/saved_models/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)

#plotting features with TSNE and saving the plot
def plot_features(model, num_classes, num_feats, batch_size, current_epoch):
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
    #plot title
    plt.title(f"TSNE plot of features at epoch {current_epoch}")
    #plot legend
    plt.legend([str(i) for i in range(num_classes)])
    #saving plot
    save_path = "./data/plots/"
    #plot name
    plt_name= f"TSNE_plot_epoch_{current_epoch}.pdf"
    plt.savefig(save_path+plt_name, format='pdf')
    #plt.show()

#plot and save losses
def plot_losses(tr_loss, val_loss,current_epoch):

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


    #Training loop

nr = 0
current_epoch = 0
epochs = 100
tr_loss = []
val_loss = []
model_name = "Kids450_test1"

for epoch in range(100):
        
    print(f"Epoch [{epoch}/{epochs}]\t",flush = True)
    stime = time.time()

    model.train()
    tr_loss_epoch = 0
    
    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.squeeze().to('cuda:0').float()
        x_j = x_j.squeeze().to('cuda:0').float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()
        
        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}",flush = True)

        tr_loss_epoch += loss.item()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    
    lr = optimizer.param_groups[0]["lr"]

    if nr == 0 and (epoch+1) % 50 == 0:
        save_model(model, optimizer, mainscheduler, current_epoch,model_name + "_checkpoint_{}.pt")

    model.eval()
    with torch.no_grad():
        val_loss_epoch = 0
        for step, (x_i, x_j) in enumerate(valid_loader):
        
          x_i = x_i.squeeze().to('cuda:0').float()
          x_j = x_j.squeeze().to('cuda:0').float()

          # positive pair, with encoding
          z_i = model(x_i)
          z_j = model(x_j)

          loss = criterion(z_i, z_j)

          if nr == 0 and step % 50 == 0:
              print(f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(),5)}",flush = True)

          val_loss_epoch += loss.item()

    if nr == 0:
        tr_loss.append(tr_loss_epoch / len(train_loader))
        val_loss.append(val_loss_epoch / len(valid_loader))
        print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}")
        print(f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(valid_loader)}\t lr: {round(lr, 5)}")
        current_epoch += 1

    dg.on_epoch_end()#reset the loss

    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes",flush = True)

    #if (epoch+1)%10==0:
    #    plot_features(model.pretrained, 10, 2048, 128,current_epoch)
    #    plot_losses(tr_loss, val_loss,current_epoch)
    #plot_features(model.pretrained, 3, 2048, 128,current_epoch)
    plot_losses(tr_loss, val_loss,current_epoch)


#save model
save_model(model, optimizer, mainscheduler, current_epoch, model_name + "_checkpoint_{}_260621.pt")

#save losses to a pickle file
with open('./data/saved_models/losses.pkl','wb') as f:
    dict = {'train_loss':tr_loss,'val_loss':val_loss}
    pickle.dump(dict,f)

print("Training completed and model saved",flush = True)
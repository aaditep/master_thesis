import numpy as np
import time, os, glob
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
from my_utils.Dataloader_class import Kids450, kids450_files_localscratch
from my_utils.NT_xent_loss import SimCLR_Loss
import yaml

#fix "too many open files error"
import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')


from my_utils.collate_file_funcs import collate_fn


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
    def __init__(self,base_model, pretrained = True):
        super().__init__()
        self.base_model = base_model
        self.pretrained = pretrained
        
        #PRETRAINED MODEL
        if self.base_model == 'resnet18':
            self.pretrained = models.resnet18(pretrained = self.pretrained)
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

def load_config(config_file):
    """Load YAML configuration from file."""
    with open(config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def prepare_datasets(file_paths : list, resolution : int):
    """
    Prepare the datasets for training and validation data

    Args:
    file_paths : list : list of file paths

    Returns:
    dg : Kids450 : dataset for training data
    vdg : Kids450 : dataset for validation data
    """
    dg = Kids450(phase = "train",file_paths = file_paths, resolution = resolution )
    vdg = Kids450(phase = "val",file_paths = file_paths, resolution = resolution)
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
                               collate_fn = collate_fn,
                               pin_memory = True,
                               num_workers = num_workers)

    valid_loader = DataLoader(vdg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = collate_fn,
                               shuffle = False,
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
        dg,
        train_data,
        valid_data,
        optimizer,
        gpu_id,
        warmupscheduler,
        mainscheduler,
        criterion,
        config,

    ) -> None: # the -> None is a return type "hint" that literally tells us that the function returns nothing
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.dataset_class = dg
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.warmupscheduler = warmupscheduler
        self.mainscheduler = mainscheduler
        self.loss = criterion
        self.save_every = config.save_every
        self.run_name = config.run_name
        self.continue_training = config.continue_training

        self.epoch_range = range(1,config.total_epochs+1)
        self.total_batches = len(self.train_data)
        self.example_ct = 0 # number of examples seen
        self.batch_ct = 0 # number of batches seen


        self.stime = 0
        self.tr_loss =[]
        self.val_loss = []
        self.tr_loss_epoch = 0
        self.val_loss_epoch = 0



    #I will try to combine the run_batch and run_epoch functions. It is too much to have them separate
    def _run_epoch(self, epoch):
        """
        Run a single epoch through the model

        Args:
        epoch : int : epoch number
        """
        b_sz = len(next(iter(self.train_data))[0]) #get batch size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        for step,(x_i,x_j) in enumerate(self.train_data):
            x_i = x_i.squeeze().to(self.gpu_id).float()
            x_j = x_j.squeeze().to(self.gpu_id).float()
            
            self.optimizer.zero_grad()
    
            z_i = self.model(x_i)
            z_j = self.model(x_j)

            loss = self.loss(z_i, z_j)
            loss.backward()

            self.example_ct += b_sz
            self.batch_ct += 1

            self.optimizer.step()
            
            if step % 50 == 0: #log loss at every 50th step
                print(f"Step [{step}/{len(self.train_data)}]\t Loss: {round(loss.item(), 5)}",flush = True)
                wandb.log({"epoch": epoch, "loss": loss}, step=self.example_ct)
            self.tr_loss_epoch += loss.item()

        if epoch < 10 and not self.continue_training:
            self.warmupscheduler.step()

        if epoch >= 10 or self.continue_training:
            self.mainscheduler.step()

        #validation cycle
        self.model.eval()
        with torch.no_grad():
            for step,(x_i,x_j) in enumerate(self.valid_data):
                x_i = x_i.squeeze().to(self.gpu_id).float()
                x_j = x_j.squeeze().to(self.gpu_id).float()
                z_i = self.model(x_i)
                z_j = self.model(x_j)
                loss = self.loss(z_i, z_j)
                
                if step % 50 == 0: #log loss at every 50th step
                    print(f"Validation Step [{step}/{len(self.valid_data)}]\t Loss: {round(loss.item(), 5)}",flush = True)
                    wandb.log({"epoch": epoch, "val_loss": loss}, step=self.example_ct)
                self.val_loss_epoch += loss.item()
        


    def train(self, max_epochs: int):
        """
        Train the model. learning rate schedulers are used for warmup and cosine decay depending on the epoch
        Args:
        max_epochs : int : maximum epochs
        """
         # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.loss, log="all", log_freq=50)
        if self.continue_training:
            self.epoch_range = range(config.last_epoch+1, config.total_epochs+1)
        for epoch in self.epoch_range:
            #epoch += 1 #start from 1
            self.tr_loss_epoch = 0
            self.val_loss_epoch = 0
            self.stime = time.time()
            self._run_epoch(epoch)
            #if epoch < 10:
            #    self.warmupscheduler.step()
            #
            #if epoch >= 10:
            #    self.mainscheduler.step()

            if epoch % self.save_every == 0:
                self.save_model(self.model, self.optimizer, self.mainscheduler , epoch, self.run_name)
                 


            self.tr_loss.append(self.tr_loss_epoch / len(self.train_data))
            self.val_loss.append(self.val_loss_epoch / len(self.valid_data))
            loss_per_epoch = self.tr_loss_epoch / len(self.train_data)
            val_per_epoch = self.val_loss_epoch / len(self.valid_data)
            time_taken = (time.time()-self.stime)/60
            print(f"Epoch [{epoch}/{max_epochs}]\t Training Loss: {loss_per_epoch}\t Time Taken: {time_taken} minutes",flush = True)
            print(f"Epoch [{epoch}/{max_epochs}]\t Validation Loss: {val_per_epoch}\t Time Taken: {time_taken} minutes",flush = True)
            wandb.log({"Time taken per epoch": time_taken, "Training loss per epoch": loss_per_epoch,"Validation loss per epoch": val_per_epoch})
            #I want to log this loss to wandb so avg loss per epoch
            self.dataset_class.on_epoch_end()
            #dg.on_epoch_end()#shuffle data inside each epoch ##TODO CHeck if this is uses it actually works

    def save_model(self,model, optimizer, scheduler, current_epoch, run_name):
        """
        Save the model

        Args:
        model : PreModel : model
        optimizer : torch.optim.Optimizer : optimizer
        scheduler : torch.optim.lr_scheduler : scheduler
        current_epoch : int : current epoch
        run_name : str : name of the model
        """
        run_name = run_name + f"_epoch_{current_epoch}.pt"
        out = os.path.join('./data/saved_models/',run_name)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}, out)



def load_model(model, optimizer, scheduler, config):
    """
    Load the model
    Returns:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    scheduler : torch.optim.lr_scheduler : scheduler
    """
    run_name = config.run_name +f"_epoch_{config.last_epoch}.pt"
    out = os.path.join('./data/saved_models/',run_name)
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler

def load_train_objs(config, file_paths):
    """
    Load the training objects

    Args:
    batch_size : int : batch sizewww
    
    Returns:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    warmupscheduler : torch.optim.lr_scheduler.LambdaLR : warmup scheduler
    mainscheduler : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts : main scheduler
    criterion : SimCLR_Loss : loss function
    """
    dg, vdg = prepare_datasets(file_paths, config.resolution)
    train_loader, valid_loader = prepare_dataloaders(dg, vdg, config.batch_size, config.num_workers)
    model = PreModel("resnet18",config.pretrained).to("cuda")
    print(f" Summary of the model: {summary(model, (4, 128, 128))}",flush=True)# Works now

    if config.continue_training:
        lr = config.last_lr
    else: 
        lr = 0.15 #default
    optimizer = LARS(
                    [params for params in model.parameters() if params.requires_grad],
                    lr=lr,
                    weight_decay=1e-6,
                    exclude_from_weight_decay=["batch_normalization", "bias"],
                    )
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True) #decay the learning rate with the cosine decay schedule without restarts"
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)#scheduler for cosine decay
    #load prevoius model
    if config.continue_training:
        model, optimizer, mainscheduler = load_model(model, optimizer, mainscheduler, config)
    
    
    criterion = SimCLR_Loss(batch_size = config.batch_size, temperature = 0.5)#loss function
    
    
    return model,dg,train_loader, valid_loader,optimizer, warmupscheduler, mainscheduler, criterion

def main(device, file_paths, config):
    """
    Main function to train the model

    Args:
    device : int : device id
    total_epochs : int : total epochs
    save_every : int : save every
    batch_size : int : batch size
    """
    model,dg,train_loader, valid_loader,optimizer, warmupscheduler, mainscheduler, criterion = load_train_objs(config,file_paths)
    trainer = Trainer(model, dg,train_loader, valid_loader, optimizer, device, warmupscheduler, mainscheduler, criterion, config)
    trainer.train(config.total_epochs)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Simclr job')
    parser.add_argument('--config', type=str, default='./config_Simclr_kids450.yaml', help='config file path')
    args = parser.parse_args()
    ######With YAML CONFIG######################
    config_path = args.config
    config = load_config(config_path)
    file_paths = kids450_files_localscratch()
    resolution = config["resolution"]
    #dg, vdg = prepare_datasets(file_paths, resolution)
    #Integrate weights and biases
    import wandb
    wandb.login()
    # tell wandb to get started
    with wandb.init(project="simclr-kids450", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config 
        device = 0  # shorthand for cuda:0
        main(device,file_paths, config)

    
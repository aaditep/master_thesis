import numpy as np
import torch
import yaml
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time, os, random
import pickle
from my_utils.cosmo_loader import  kids450_files_cluster, kids450_files_localscratch, regr_Kids450
from my_utils.cosmo_collate import cosmotrain_collate_fn
from torchsummary import summary
import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from my_utils.models import custom_pretrainingmodel, custom_DSModel, custom_Regression_model
from my_utils.real_resnets import Resnet_regressionmodel




def load_config(config_file):

    """Load YAML configuration from file.
    args:
    config_file : str : path to config file
    returns:
    config : dict : configuration dictionary
    
    """
    with open(config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def prepare_datasets(file_paths_train : list, file_paths_test : list, resolution : int):
    """
    Prepare the datasets for training and validation data

    Args:
    file_paths_train : list : list of file paths for training data
    file_paths_test : list : list of file paths for test data
    resolution : int : resolution of the data

    Returns:
    dg : Kids450 : dataset for training data
    vdg : Kids450 : dataset for validation data
    tdg : Kids450 : dataset for test data
    """
    dg = regr_Kids450(phase = "train",file_paths = file_paths_train,resolution = resolution)
    vdg = regr_Kids450(phase = "val",file_paths = file_paths_train,resolution = resolution)
    tdg =  regr_Kids450(phase = "test",file_paths = file_paths_test,resolution = resolution)
    return dg, vdg, tdg


def prepare_dataloaders(dg, vdg, tdg, batch_size: int, num_workers: int):
    """
    Prepare the dataloaders for training and validation data

    Args:
    dg : Kids450 : dataset for training data
    vdg : Kids450 : dataset for validation data
    tdg : Kids450 : dataset for test data
    batch_size : int : batch size
    num_workers : int : number of workers for dataloader    

    Returns:
    train_loader : DataLoader : dataloader for training data
    valid_loader : DataLoader : dataloader for validation data   
    test_loader : DataLoader : dataloader for test data
    """
    train_loader = DataLoader(dg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = cosmotrain_collate_fn,
                               pin_memory = True,
                               num_workers = num_workers)

    valid_loader = DataLoader(vdg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = cosmotrain_collate_fn,
                               shuffle = False,
                               pin_memory = True,
                               num_workers = num_workers)

    test_loader = DataLoader(tdg,batch_size = batch_size,
                               drop_last=True,
                               prefetch_factor = 3,
                               collate_fn = cosmotrain_collate_fn,
                               shuffle = False,
                               pin_memory = True,
                               num_workers = num_workers)
    return train_loader, valid_loader, test_loader



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
        test_data,
        optimizer,
        device,
        mainscheduler,
        criterion,
        config,

    ) -> None: # the -> None is a return type "hint" that literally tells us that the function returns nothing
        self.device = device
        self.model = model.to(self.device)
        self.dataset_class = dg
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = mainscheduler
        self.loss = criterion
        self.save_every = config.save_every #save model every x epochs
        self.run_name = config.run_name #name of the model and run that will be saved
        self.continue_training = config.continue_training # bool for continue training from a previous model

        self.epoch_range = range(1,config.total_epochs+1) #range of epochs
        self.total_batches = len(self.train_data) #total number of batches
        self.example_ct = 0 # number of examples seen
        self.batch_ct = 0 # number of batches seen


        self.stime = 0  #start time
        self.tr_loss =[] #training loss WITHIN epoch
        self.val_loss = []  #validation loss WITHIN epoch
        self.tr_loss_epoch = 0 #training loss OVER epoch
        self.val_loss_epoch = 0 #validation loss OVER epoch



    #I will try to combine the run_batch and run_epoch functions. It is too much to have them separate
    def _run_epoch(self, epoch):
        """
        Run a single epoch through the model

        Args:
        epoch : int : epoch number
        """
        b_sz = len(next(iter(self.train_data))[0]) #get batch size
        print(f"[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.model.train()
        for step,(x,y) in enumerate(self.train_data):      
            self.optimizer.zero_grad()

            data = x.squeeze().to(self.device).float()
            target = y.squeeze().to(self.device).float()
    
            output = self.model(data)

            loss = self.loss(output, target)
            loss.backward()

            self.optimizer.step()

            self.example_ct += b_sz
            self.batch_ct += 1
            
            if step % 50 == 0: #log loss at every 50th step
                print(f"Step [{step}/{len(self.train_data)}]\t Loss: {round(loss.item(), 5)}",flush = True)
                wandb.log({"epoch": epoch, "loss": loss}, step=self.example_ct)
            self.tr_loss_epoch += loss.item()

        #validation cycle
        self.model.eval()
        with torch.no_grad():

            for step,(x,y) in enumerate(self.valid_data):      
                self.optimizer.zero_grad()

                data = x.squeeze().to(self.device).float()
                target = y.squeeze().to(self.device).float()

                output = self.model(data)

                loss = self.loss(output, target)
                
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

            #shceduler  step
            #self.scheduler.step()

            if epoch % self.save_every == 0 or epoch == 1:
                self.save_model(self.model, self.optimizer, self.scheduler , epoch, self.run_name)
                 


            self.tr_loss.append(self.tr_loss_epoch / len(self.train_data))
            self.val_loss.append(self.val_loss_epoch / len(self.valid_data))
            loss_per_epoch = self.tr_loss_epoch / len(self.train_data)
            val_per_epoch = self.val_loss_epoch / len(self.valid_data)

            time_taken = (time.time()-self.stime)/60
            print(f"Epoch [{epoch}/{max_epochs}]\t Training Loss: {loss_per_epoch}\t Time Taken: {time_taken} minutes",flush = True)
            print(f"Epoch [{epoch}/{max_epochs}]\t Validation Loss: {val_per_epoch}\t Time Taken: {time_taken} minutes",flush = True)
            print("Lr per epoch:", self.optimizer.param_groups[0]["lr"],flush = True)
            #shceduler  step
            self.scheduler.step(val_per_epoch)            
            
            wandb.log({"Time taken per epoch": time_taken, "Training loss per epoch": loss_per_epoch,"Validation loss per epoch": val_per_epoch, "Lr per epoch " :self.optimizer.param_groups[0]["lr"]})
            #I want to log this loss to wandb so avg loss per epoch
            self.dataset_class.on_epoch_end()
            #dg.on_epoch_end()#shuffle data inside each epoch ##TODO CHeck if this is uses it actually worksss


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

        save_path = "/cluster/work/refregier/atepper/saved_models/" + run_name +"/"  #work storage
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        run_name = run_name + f"_epoch_{current_epoch}.pt"
        #save_path = './data/saved_models/'

        if self.continue_training:
            run_name = run_name + f"_epoch_{current_epoch + config.last_epoch}.pt"  
        out = os.path.join(save_path,run_name)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()}, out)
        
def load_model(model, optimizer, scheduler, config):
    """
    Load the model
    args:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    scheduler : torch.optim.lr_scheduler : scheduler
    config : dict : configuration dictionary
    Returns:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    scheduler : torch.optim.lr_scheduler : scheduler
    """
    run_name = config.run_name +f"_epoch_{config.last_epoch}.pt"
    #save_path = './data/saved_models/'
    save_path = "/cluster/work/refregier/atepper/saved_models/" #work storage
    out = os.path.join(save_path ,run_name)
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler

def load_train_objs(config, file_paths_train, file_paths_test):
    """
    Load the training objects

    Args:
    config : dict : configuration dictionary
    file_paths_train : list : list of file paths for training data
    file_paths_test : list : list of file paths for test data
    
    Returns:
    model : PreModel : model
    dg : Kids450 : dataset for training data
    train_loader : DataLoader : dataloader for training data
    valid_loader : DataLoader : dataloader for validation data
    optimizer : torch.optim.Optimizer : optimizer
    mainscheduler : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts : main scheduler
    criterion : SimCLR_Loss : loss function
    """
    dg, vdg, tdg = prepare_datasets(file_paths_train, file_paths_test, config.resolution)
    train_loader, valid_loader,test_loader = prepare_dataloaders(dg, vdg, tdg, config.batch_size, config.num_workers)


    #### stock Resnet model
    model = Resnet_regressionmodel('resnet34',
                                   pretrained_weights = True, 
                                   dropout_rate = config.dropout, 
                                   head_type = config.projection_head).to('cuda:0')
    #### stock Resnet model

    ##### custom resnet model
    #
    #model = custom_Regression_model("custom_resnet",
    #                               layers = [3, 4, 6, 3], 
    #                               dropout_rate = config.dropout, 
    #                               head_type = config.projection_head).to('cuda:0')
    ###### custom resnet model


    print(f" Summary of the model: {summary(model, (4, 128, 128))}",flush=True)
    print(f'Model dropout rate: {config.dropout}',flush=True)
    lr = config.learning_rate   #0.5e-4 #default
    optimizer = optim.Adam(model.parameters(),lr = lr, betas = (0.9, 0.999))
    mainscheduler =  ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=5.0e-7)
    #mainscheduler =  StepLR(optimizer, step_size=10, gamma=0.1) # just a dummy scheduler
    #load prevoius model
    if config.continue_training:
        model, optimizer, mainscheduler = load_model(model, optimizer, mainscheduler, config)
    
    criterion = nn.MSELoss()#loss function of MSE
    
    
    return model,dg,train_loader, valid_loader, test_loader, optimizer, mainscheduler, criterion




def main(device, file_paths_train, file_paths_test, config):
    """
    Main function to train the model

    Args:
    device : int : device id
    file_paths_train : list : list of file paths for training data
    file_paths_test : list : list of file paths for test data
    config : dict : configuration dictionary
    """
    model,dg,train_loader, valid_loader,test_loader,optimizer, mainscheduler, criterion = load_train_objs(config,file_paths_train, file_paths_test)
    trainer = Trainer(model, dg,train_loader, valid_loader, test_loader, optimizer, device, mainscheduler, criterion, config)
    trainer.train(config.total_epochs)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='Regression job')
    parser.add_argument('--config', type=str, default='./config_regr_kids450.yaml', help='config file path')
    args = parser.parse_args()
    ######With YAML CONFIG######################
    config_path = args.config
    config = load_config(config_path)
    file_paths_train, file_paths_test = kids450_files_localscratch()
    resolution = config["resolution"]
    #dg, vdg = prepare_datasets(file_paths, resolution)
    #Integrate weights and biases
    wandb.login()
    # tell wandb to get started
    with wandb.init(project="regression-kids450", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config 
        device = 0  # shorthand for cuda:0
        main(device,file_paths_train, file_paths_test, config)

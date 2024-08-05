import numpy as np
import math
import time, os, random
import pickle
from tqdm import tqdm
from functools import partial
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import wandb

import torch
import torch.nn as nn
import torchvision
#import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage,ToTensor, Compose, Normalize,Resize
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

from my_utils.mae_dataloader import  kids450_files_cluster, mae_kids450, kids450_files_localscratch
from my_utils.mae_collate import mae_collate_fn
from my_utils.mae_base import MaskedAutoencoderViT

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block
from my_utils.pos_embeds import get_2d_sincos_pos_embed
#comment 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(config):
    setup_seed(config.seed)

    batch_size = config.batch_size
    load_batch_size = min(config.max_device_batch_size, batch_size)
    print("Using batch size", load_batch_size, flush=True)

    assert batch_size % load_batch_size == 0
    steps_per_update = 1 #batch_size // load_batch_size

    #train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(),Resize((224, 224)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    #batch_size = 128
    resolution = config.img_size
    train_data = mae_kids450(phase = "train",file_paths = file_paths_train,resolution = resolution)
    #valid_data = mae_kids450(phase = "val",file_paths = file_paths_train,resolution = resolution)
    kids_train_loader = DataLoader(train_data,batch_size = load_batch_size , drop_last=True,collate_fn = mae_collate_fn, num_workers = 19)
    #kids_valid_loader = DataLoader(valid_data,batch_size = load_batch_size , drop_last=True,collate_fn = mae_collate_fn, num_workers = 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = mae_vit_base_patch8(config).to(device)
    print(model)
    #lr=config.base_learning_rate * config.batch_size / 256
    #optim = torch.optim.AdamW(model.parameters(), lr=config.base_learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    #lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
    #lr_func = lambda epoch: 0.005 * (math.cos(epoch / 2000* math.pi) +1 )
    #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    optim = torch.optim.Adam(model.parameters(),lr = config.base_learning_rate, betas = (0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.8, patience=5, min_lr=5.0e-7)


    #########for_training_cont
    if config.continue_previous_train == True:
        print("loading model", flush = True)
        model, optim, lr_scheduler = load_model(model, optim, lr_scheduler, config)
        start_epoch = config.last_epoch -1 # minus one becasue the indexes start from 0
        end_epoch = config.total_epoch + config.last_epoch
    else:
        start_epoch = 0 
        end_epoch = config.total_epoch -1# minus one becasue the indexes start from 0
        
    epoch_range = range(start_epoch, end_epoch)
        

    start_lr = optim.param_groups[0]["lr"]
    print(f"Starting from epoch {start_epoch +1} and planning to finish at {end_epoch}",flush = True)
    print(f"Starting with Lr per epoch  {start_lr}",flush = True)



    
    wandb.watch(model, log="all", log_freq=100)
    step_count = 0
    optim.zero_grad()
    for e in epoch_range:
        model.train()
        #if e == 3:
        #    break
        stime = time.time()
        losses = []
        for img,label in tqdm(iter(kids_train_loader)):
            step_count += 1
            img = img.to(device)# images to gpu
            loss, _, _ = model(img, mask_ratio=config.mask_ratio)
            wandb.log({"epoch": e+1, "loss": loss}, step=step_count)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        #lr_scheduler.step()
        time_taken = (time.time()-stime)/60
        avg_loss = sum(losses) / len(losses)
        lr_scheduler.step(avg_loss)
        #writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        wandb.log({"Time taken per epoch": time_taken,
           "Training loss per epoch": avg_loss,
           #"Validation loss per epoch": val_per_epoch,
           "Lr per epoch " : optim.param_groups[0]["lr"]})
        
        train_data.on_epoch_end()#randomize data after an epoch

        ''' save model '''
        #save_model(model, optimizer, scheduler, current_epoch, run_name)
        if e % config.save_every == 0 or e == 1:
            save_model(model, optim, lr_scheduler , e, config.run_name)

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
    save_path = "/cluster/work/refregier/atepper/saved_models_mae/" + config.run_name +"/" #work storage
    out = os.path.join(save_path ,run_name)
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler


def save_model(model, optimizer, scheduler, current_epoch, run_name):
    """
    Save the model
    Args:
    model : PreModel : model
    optimizer : torch.optim.Optimizer : optimizer
    scheduler : torch.optim.lr_scheduler : scheduler
    current_epoch : int : current epoch
    run_name : str : name of the model
    """
    save_path = "/cluster/work/refregier/atepper/saved_models_mae/" + run_name +"/"  #work storage
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    run_name = run_name + f"_epoch_{current_epoch}.pt"
    #save_path = './data/saved_models/'
    out = os.path.join(save_path,run_name)
    print(f"saving as {out}", flush = True)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)



def mae_vit_base_patch8(config):
    """inputs with config file"""
    model = MaskedAutoencoderViT(
        img_size = config.img_size, patch_size = config.patch_size, in_chans = config.in_chans,
        embed_dim = config.embed_dim, depth = config.depth, num_heads = config.num_heads,
        decoder_embed_dim = config.decoder_embed_dim, decoder_depth = config.decoder_depth,
        decoder_num_heads = config.decoder_num_heads, mlp_ratio = config.mlp_ratio,
        norm_layer = partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


config = {
    "seed" : 62,
    "batch_size" : 1024,#4096,
    "max_device_batch_size" : 128,#512,
    "mask_ratio" : 0.5,
    "base_learning_rate": 1e-3,
    "weight_decay" : 0.01,#0.05
    "total_epoch" : 500,
    "warmup_epoch" : 10,
    "run_name" : "mae_huge_normed_75_patch8",
    "img_size" : 128,
    "patch_size": 8,
    "in_chans" : 4,
    "embed_dim" : 768,
    "depth" : 12,
    "num_heads" : 12,
    "decoder_embed_dim" : 512,
    "decoder_depth" : 8,
    "decoder_num_heads" : 16,
    "mlp_ratio" : 4.,
    "save_every" : 5,
    "norm_pix_loss" : False,
    "continue_previous_train" : False,
    "last_epoch" : 0,
    }


kwargs = {
    "proj_drop" : 0.2,
    "attn_drop" : 0.2,

}

if __name__ == "__main__":
    wandb.login()
    # tell wandb to get started
    with wandb.init(project="new_masked_auto_meta", config=config):
        os.environ['WANDB_MODE'] = 'online'
        # access all HPs through wandb.config, so logging matches execution!
        file_paths_train, file_paths_test = kids450_files_localscratch()
        #if wandb.run.mode == 'online':
        #    print("wandb is online")
        #else:
        #    print("wandb is offline")
        config = wandb.config 
        device = 0  # shorthand for cuda:0
        main(config)


#import argparse
#    parser = argparse.ArgumentParser(description='Simclr pretraining')
#    parser.add_argument('--config', type=str, default='./config_regr_kids450.yaml', help='config file path')
#    args = parser.parse_args()
#    ######With YAML CONFIG######################
#    config_path = args.config
#    config = load_config(config_path)
#
#def load_config(config_file):
#
#    """Load YAML configuration from file.
#    args:
#    config_file : str : path to config file
#    returns:
#    config : dict : configuration dictionary
#    
#    """
#    with open(config_file, "r") as yaml_file:
#        config = yaml.safe_load(yaml_file)
#    return config        '



#class MAE_ViT(torch.nn.Module):
#    def __init__(self,
#                 image_size=32,
#                 patch_size=2,
#                 emb_dim=192,
#                 encoder_layer=12,
#                 encoder_head=3,
#                 decoder_layer=4,
#                 decoder_head=3,
#                 mask_ratio=0.75,

#def mae_vit_huge_patch14_dec512d8b(**kwargs):
#    model = MaskedAutoencoderViT(
#        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#    return model




#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 1024,#512,
#    "mask_ratio" : 0.75,
#    "base_learning_rate": 1e-3,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_tiny_normed_75_plat",
#    "img_size" : 128,
#    "patch_size": 16,
#    "in_chans" : 4,
#    "embed_dim" : 256,
#    "depth" : 12,
#    "num_heads" : 4,
#    "decoder_embed_dim" : 256,
#    "decoder_depth" : 4,
#    "decoder_num_heads" : 4,
#    "mlp_ratio" : 4.,
#    "save_every" : 5,
#    "norm_pix_loss" : False,
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }

#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 512,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-3,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_huge_normed_50_platou",
#    "img_size" : 128,
#    "patch_size": 16,
#    "in_chans" : 4,
#    "embed_dim" : 768,
#    "depth" : 12,
#    "num_heads" : 12,
#    "decoder_embed_dim" : 512,
#    "decoder_depth" : 8,
#    "decoder_num_heads" : 16,
#    "mlp_ratio" : 4.,
#    "save_every" : 5,
#    "norm_pix_loss" : False,
#    "continue_previous_train" : True,
#    "last_epoch" : 55,
#    }
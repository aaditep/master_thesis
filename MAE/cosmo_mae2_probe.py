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

from my_utils.mae_dataloader_lin_probe import  kids450_files_cluster, mae_kids450, kids450_files_localscratch
from my_utils.mae_collate import mae_collate_fn
#from my_utils.mae_base2 import MaskedAutoencoderViT_ver2#Exchanged this!!!!
from my_utils.mae_base3 import MaskedAutoencoderViT_ver2

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block
from my_utils.pos_embeds import get_2d_sincos_pos_embed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ViT_regresser(torch.nn.Module):
    def __init__(self, encoder, embed_dim, head_type = "linear" ,head_dropout = 0.0, num_classes=2) -> None:
        super().__init__()
        #maybe I can cancel them here
        #for p in encoder.parameters():
        #    p.requires_grad = False
        self.head_dropout = head_dropout
        self.embed_dim = embed_dim
        self.patch_embed = encoder.patch_embed
        self.cls_token = encoder.cls_token
        self.pos_embed = encoder.pos_embed
        self.pos_drop = nn.Dropout(p=0.2)
        print(self.pos_embed.shape)
        self.patchify = patchify
        self.blocks = encoder.blocks
        self.norm = encoder.norm
        #print(self.pos_embed.shape)
        #self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)
        if head_type == 'linear':
            self.projector = torch.nn.Linear(self.pos_embed.shape[-1], num_classes)
        if head_type == 'non_linear':
            self.projector = nn.Sequential(
                torch.nn.Linear(self.pos_embed.shape[-1],128),
                #torch.nn.BatchNorm1d(128), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(128,num_classes))
        if head_type == 'non_linear_big':
            self.projector = nn.Sequential(
                torch.nn.Linear(self.pos_embed.shape[-1],128),
                torch.nn.BatchNorm1d(128), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(128,64),
                torch.nn.BatchNorm1d(64), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(64,2))
        if head_type == 'non_linear_big_big':
            self.projector = nn.Sequential(
                torch.nn.Linear(self.pos_embed.shape[-1],128),
                torch.nn.BatchNorm1d(128), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(128,64),
                torch.nn.BatchNorm1d(64), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(64,32),
                torch.nn.BatchNorm1d(32), #added batchnorm
                nn.ReLU(),
                nn.Dropout(self.head_dropout),
                torch.nn.Linear(32,2))
            
            
        
        self.global_pool = False
        if self.global_pool:
            norm_layer = nn.LayerNorm
            embed_dim = self.embed_dim
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
            
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
            outcome = self.projector(outcome)
            
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            outcome = self.projector(outcome)
        return outcome
    
def load_pretrain_model(model, config):

    """
    Load the pretrained model only
    args:
    model : PreModel : model
    config : dict : configuration dictionary
    Returns:
    model : PreModel : model

    """
    run_name = config.pretrain_run_name +f"_epoch_{config.pretrain_checkpoint}.pt"
    #save_path = './data/saved_models/'
    save_path = "/cluster/work/refregier/atepper/saved_models_mae/" + config.pretrain_run_name +"/" #work storage
    out = os.path.join(save_path ,run_name)
    checkpoint = torch.load(out)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


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
    print("loading state dict from: ", out)
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
    



def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    c = imgs.shape[1]
    p = 16
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))

    return x


def main(config):
    setup_seed(config.seed)

    batch_size = config.batch_size
    load_batch_size = min(config.max_device_batch_size, batch_size)
    print("Using batch size", load_batch_size, flush=True)

    assert batch_size % load_batch_size == 0
    steps_per_update = 1 #batch_size // load_batch_size

    resolution = config.img_size
    #file_paths_train, file_paths_test = kids450_files_localscratch()
    train_data = mae_kids450(phase = "train",file_paths = file_paths_train,resolution = resolution)
    valid_data = mae_kids450(phase = "val",file_paths = file_paths_train,resolution = resolution)
    kids_train_loader = DataLoader(train_data,batch_size = load_batch_size , drop_last=True,collate_fn = mae_collate_fn, num_workers = 19)
    kids_valid_loader = DataLoader(valid_data,batch_size = load_batch_size , drop_last=True,collate_fn = mae_collate_fn, num_workers = 19)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained_model = mae_vit_base_patch8(config).to(device)
    #encoder = model.encoder
    #vit_regr = ViT_regresser(encoder,config.embed_dim).to(device)
    print(pretrained_model)

    if config.training_type == "lin_probe":
        print("loading pretrained model checkpoints for linear probing", flush = True)
        pretrained_model = load_pretrain_model(pretrained_model,config)
        encoder = pretrained_model.encoder
        model = ViT_regresser(encoder,config.embed_dim,head_type = config.head_type, head_dropout = config.head_dropout).to(device)
        print(pretrained_model)

        #for name, p in vit_regr.named_parameters():
        #    print(f'{name}: requires_grad={p.requires_grad}')
        #freeze params and unfreeze head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.projector.named_parameters():
            p.requires_grad = True
    if config.training_type == "vit_scratch":
        #if not linear probing, just use the model as is training from scratch
        encoder = pretrained_model.encoder
        model = ViT_regresser(encoder,config.embed_dim,head_type = config.head_type, head_dropout = config.head_dropout).to(device)
        print("Initiating pure vision transformer from scratch", flush = True)
        #print(load_pretrain_model)
    if config.training_type == "fine_tune":
        print("loading pretrained model checkpoints for finetuning", flush = True)
        pretrained_model = load_pretrain_model(pretrained_model,config) #maybe this somehow confuses it.
        encoder = pretrained_model.encoder
        model = ViT_regresser(encoder,config.embed_dim,head_type = config.head_type, head_dropout = config.head_dropout).to(device)
        #print(load_pretrain_model)

    print("initializing optimizer with model",model, flush = True)

    optim = torch.optim.Adam(model.parameters(),lr = config.base_learning_rate, betas = (0.9, 0.999))
    print("initializing lr scheduler", flush = True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.8, patience=5, min_lr=5.0e-7)
    print("initializing loss function", flush = True)
    calc_loss = nn.MSELoss()

    

    if config.continue_previous_train == True:
        print("continue train", flush = True)
        model, optim, lr_scheduler = load_model(model, optim, lr_scheduler, config)
        start_epoch = config.last_epoch 
        end_epoch = config.total_epoch + config.last_epoch

    else:
        start_epoch = 0 
        end_epoch = config.total_epoch -1# minus one becasue the indexes start from 0
        
    epoch_range = range(start_epoch, end_epoch)
    
    
    start_lr = optim.param_groups[0]["lr"]
    print(f"Starting model:" ,model,flush = True)
    print(f"Starting from epoch {start_epoch +1} and planning to finish at {end_epoch}",flush = True)
    print(f"Starting with Lr per epoch  {start_lr}",flush = True)


    wandb.watch(model, log="all", log_freq=100)
    step_count = 0
    for e in epoch_range:
        model.train()
        optim.zero_grad()
        #if e == 3:
        #    break
        stime = time.time()
        losses = []
        for img,label in tqdm(iter(kids_train_loader)):
            step_count += 1
            img = img.to(device)# images to gpu
            target = label.to(device)
            pred = model(img)
            loss = calc_loss(pred, target)     

            wandb.log({"epoch": e+1, "loss": loss}, step=step_count)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        time_taken = (time.time()-stime)/60
        avg_loss = sum(losses) / len(losses)
        lr_scheduler.step(avg_loss)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for img,label in tqdm(iter(kids_valid_loader)):
                img = img.to(device)# images to gpu
                target = label.to(device)
                pred = model(img)
                loss = calc_loss(pred, target) 
                val_losses.append(loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)


        print(f'In epoch {e}, average traning loss is {avg_loss}.')
        print(f'In epoch {e}, average validation loss is {avg_val_loss}.')
        wandb.log({"Time taken per epoch": time_taken,
           "Training loss per epoch": avg_loss,
           "Validation loss per epoch": avg_val_loss,
           "Lr per epoch " : optim.param_groups[0]["lr"]})
        
        train_data.on_epoch_end()#randomize data after an epoch

        ''' save model '''
        #save_model(model, optimizer, scheduler, current_epoch, run_name)
        if e % config.save_every == 0 or e == 1:
            #print(f"would save as {config.run_name}")
            save_model(model, optim, lr_scheduler , e, config.run_name)

def mae_vit_base_patch8(config):
    """inputs with config file"""
    model = MaskedAutoencoderViT_ver2(
        img_size = config.img_size, patch_size = config.patch_size, in_chans = config.in_chans,
        embed_dim = config.embed_dim, depth = config.depth, num_heads = config.num_heads,
        decoder_embed_dim = config.decoder_embed_dim, decoder_depth = config.decoder_depth,
        decoder_num_heads = config.decoder_num_heads, mlp_ratio = config.mlp_ratio,
        norm_layer = partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

config = {
    "seed" : 62,
    "batch_size" : 1024,#4096,
    "max_device_batch_size" : 2048,#512,
    "mask_ratio" : 0.5,
    "base_learning_rate": 1e-4,
    "weight_decay" : 0.01,#0.05
    "total_epoch" : 500,
    "warmup_epoch" : 10,
    "run_name" : "mae_base_224_normed_mae_base3_non_linear_probing",
    "img_size" : 224,
    "patch_size": 16,
    "in_chans" : 4,
    "embed_dim" : 256,
    "depth" : 12,
    "num_heads" : 4,
    "decoder_embed_dim" : 256,
    "decoder_depth" : 4,
    "decoder_num_heads" : 4,
    "mlp_ratio" : 4.,
    "save_every" : 5,
    "norm_pix_loss" : False,
    "head_type" : 'non_linear',
    "head_dropout" : 0.2,
    "training_type" : "lin_probe",
    "pretrain_run_name" : "mae_base_224_normed_sml_model_mae2_mae_base3_2",
    "pretrain_checkpoint" : 215, #if continue, be sure to change this
    "continue_previous_train" : False,
    "last_epoch" : 0,
    }#
kwargs = {
    "proj_drop" : 0.1,
    "attn_drop" : 0.1,

}

if __name__ == "__main__":
    wandb.login()
    # tell wandb to get started
    with wandb.init(project="new_masked_auto_linprobe", config=config):
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
#vit scratch
#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 512,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-4,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_non_linear_probe",
#    "img_size" : 224,
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
#    "head_type" : "non_linear",
#    "head_dropout" : 0.3,
#    "training_type" : "vit_scratch",
#    "pretrain_run_name" : "mae_base_224_normed_sml_model_mae2",
#    "pretrain_checkpoint" : 115, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }#


#fine tune
#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 512,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-4,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_fine_tune_full_non_linear_probe_try2",
#    "img_size" : 224,
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
#    "head_type" : "non_linear",
#    "head_dropout" : 0.3,
#    "training_type" : "fine_tune",
#    "pretrain_run_name" : "mae_base_224_normed_sml_model_mae2",
#    "pretrain_checkpoint" : 230, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }
#linear probe
#config = {
#    "seed" : 62,
#    "batch_size" : 4096,#4096,
#    "max_device_batch_size" : 4096,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-2,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_non_lin_test_big_big_head2_4096real",
#    "img_size" : 224,
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
#    "head_type" : "non_linear_big_big",
#    "head_dropout" : 0.1,
#    "training_type" : "lin_probe",
#    "pretrain_run_name" : "mae_base_224_normed_sml_model_mae2",
#    "pretrain_checkpoint" : 230, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }


#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 512,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-4,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae__non_linear_probe_128pix",
#    "img_size" : 128,
#    "patch_size": 16,
#    "in_chans" : 4,
#    "embed_dim" : 768,
#    "depth" : 12,
#    "num_heads" : 12,
#    "decoder_embed_dim" : 512,
#    "decoder_depth" : 8,
#    "decoder_num_heads" : 8,
#    "mlp_ratio" : 4.,
#    "save_every" : 5,
#    "norm_pix_loss" : False,
#    "head_type" : "non_linear",
#    "head_dropout" : 0.2,
#    "training_type" : "vit_scratch",
#    "pretrain_run_name" : "none",
#    "pretrain_checkpoint" : 0, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }#
#kwargs = {
#    "proj_drop" : 0.1,
#    "attn_drop" : 0.1,
#
#}

#####8patchfinetune

#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 256,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-4,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae_non_linear_probe_8_finetune",
#    "img_size" : 128,
#    "patch_size": 8,
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
#    "head_type" : "non_linear",
#    "head_dropout" : 0.2,
#    "training_type" : "finetune",
#    "pretrain_run_name" : "mae_patch8_test_again",
#    "pretrain_checkpoint" : 200, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }#
#kwargs = {
#    "proj_drop" : 0.1,
#    "attn_drop" : 0.1,
#
#}
#
#########8patchsscratch
#config = {
#    "seed" : 62,
#    "batch_size" : 1024,#4096,
#    "max_device_batch_size" : 256,#512,
#    "mask_ratio" : 0.5,
#    "base_learning_rate": 1e-4,
#    "weight_decay" : 0.01,#0.05
#    "total_epoch" : 500,
#    "warmup_epoch" : 10,
#    "run_name" : "mae__non_linear_probe_8_patch",
#    "img_size" : 128,
#    "patch_size": 8,
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
#    "head_type" : "non_linear",
#    "head_dropout" : 0.2,
#    "training_type" : "vit_scratch",
#    "pretrain_run_name" : "none",
#    "pretrain_checkpoint" : 0, #if continue, be sure to change this
#    "continue_previous_train" : False,
#    "last_epoch" : 0,
#    }#
#kwargs = {
#    "proj_drop" : 0.1,
#    "attn_drop" : 0.1,
#
#}
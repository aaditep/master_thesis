from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from my_utils.pos_embeds import get_2d_sincos_pos_embed




class MaskedAutoencoderViT_ver2(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=4,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,**kwargs):
        super().__init__()

        #global stuff
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.encoder = Encoder(self.patch_embed, num_patches, embed_dim, num_heads, mlp_ratio, norm_layer, depth, **kwargs)
        self.decoder = Decoder(patch_size, in_chans, num_patches, embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer, decoder_depth, **kwargs)

        


        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.encoder.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.encoder.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.encoder.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, mask_ratio):
        x, mask, ids_restore = self.encoder(x, mask_ratio)
        x = self.decoder(x, ids_restore)
        return x, mask
        





class Encoder(nn.Module):
    def __init__(self, patch_embed, num_patches, embed_dim,num_heads, mlp_ratio, norm_layer, depth, **kwargs):
        super().__init__()
        #define for encoder: 
        self.patch_embed = patch_embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #random masking#########
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,**kwargs)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        #input needed patch_embed

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        #here add the stuff to change the model only to train for the params

        return x, mask,ids_restore
    


class Decoder(nn.Module):
    def __init__(self, patch_size, in_chans, num_patches, embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer, decoder_depth, **kwargs):
        super().__init__()
        #define for encoder: 
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,**kwargs)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2*in_chans, bias = True)




    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
class mae_DSModel(nn.Module):
    def __init__(self,mae_model,embed_dim, dropout_rate, head_type):
        super().__init__()
        
        self.premodel = mae_model
        self.dropout_rate = dropout_rate
        self.head_type = head_type
        self.embed_dim = embed_dim
        
        #set rquieres grad to false for the premodel to avoid training the main body of it it
        for p in self.premodel.encoder.parameters():
            p.requires_grad = False
            
        for p in self.premodel.decoder.parameters():
            p.requires_grad = False
        
        
        if self.head_type == "linear_head":
            self.projector = nn.Linear(self.embed_dim, 2)
            self.fc_norm = nn.LayerNorm(self.embed_dim)
        #if self.head_type == 'non_linear_head':
        #    self.projector = 
        
    def forward(self,x):
        """
        By not adding here premodel.projector you are omitting it
        """
        out = self.premodel.encoder(x)
        #omit the premodel projector and replace with the needed new projector
        out = self.fc_norm(out)
        out = self.projector(out)
        return out
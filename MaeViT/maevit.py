import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from helpers import *
from modules import *

class MaeViT(nn.Module):
    def __init__(self, *, image_size, patch_size, 
                 enc_dim, dec_dim, enc_depth, dec_depth, enc_heads, dec_heads, 
                 mlp_factor, channels = 3, dim_head = 64):
        super().__init__()

        patch_height, patch_width, patch_dim, self.h, self.w = get_shapes(image_size, patch_size, channels)

        self.enc_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, enc_dim),
            nn.LayerNorm(enc_dim))
        
        self.encoder = Transformer(enc_dim, enc_depth, enc_heads, dim_head, mlp_factor)
        self.encoder_norm = nn.LayerNorm(enc_dim)
        
        self.dec_patch_embed = nn.Linear(enc_dim, dec_dim)
        self.decoder = Transformer(dec_dim, dec_depth, dec_heads, dim_head, mlp_factor)
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, 3*patch_height*patch_width)
        
        self.unpatch = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=self.h, w=self.w)
        
    def forward(self, x):
        *_, h, w, dtype = *x.shape, x.dtype
        x = self.enc_patch_embed(x)
        x = x + posemb_sincos_2d(x, self.h, self.w)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        x = self.dec_patch_embed(x) 
        x = x + posemb_sincos_2d(x, self.h, self.w)
        x = self.decoder(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return self.unpatch(x)

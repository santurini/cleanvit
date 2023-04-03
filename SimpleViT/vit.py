import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from helpers import *
from modules import *

class CleanViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3 * image_height * image_width)
        )

        self.rearrange = Rearrange('b (c h w) ->  b c h w', h = image_height, w = image_width)
        
    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype
        
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)
        
        x = self.linear_head(x)
        x = self.rearrange(x)
        return x

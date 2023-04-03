import warnings
warnings.filterwarnings("ignore")

import torch
import wandb
import argparse
from vit import CleanViT
from lightning import LightningViT
from dataset import Compressed
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=False, default='gpu', help='training device (cpu/gpu)')
parser.add_argument('--cpu_w', type=int, required=False, default=0, help='number of cpu workers in dataloader')
parser.add_argument('--path', type=str, required=True, help='input folder path')
parser.add_argument('--split', type=float, required=False, default=0.8, help='train/val split')
parser.add_argument('--bs', type=int, required=False, default=4, help='batch size')
parser.add_argument('--epochs', type=int, required=False, default=50, help='train epochs')
parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='path to ckeckpoint')
parser.add_argument('--image_size', type=int, required=False, default=256, help='input image size')
parser.add_argument('--patch_size', type=int, required=False, default=16, help='vit patch size')
parser.add_argument('--depth',  type=int, required=False, default=6, help='transformer blocks')
parser.add_argument('--heads', type=int, required=False, default=8, help='number of heads')
parser.add_argument('--dim', type=int, required=False, default=512, help='hidden dimension')
parser.add_argument('--mlp_dim', type=int, required=False, default=1024, help='mlp hidden dimension')
parser.add_argument('--channels', type=int, required=False, default=3, help='input channels')
parser.add_argument('--dim_head', type=int, required=False, default=64, help='head dimensionality')
args = parser.parse_args()

wandb.login(key='191e81893a41f570331354ae4c2aa8e99a4bba48')    
logger = WandbLogger(project='video-super-resolution')

print('\nPREPARING DATALOADERS...')
tr_ds = Compressed(args.path, args.image_size, 'train', args.split)
val_ds = Compressed(args.path, args.image_size, 'val', args.split)
tr_dl = DataLoader(tr_ds,  args.bs, True, num_workers=args.cpu_w)
val_dl = DataLoader(val_ds,  args.bs, False, num_workers=args.cpu_w)

print('\nINITIALIZING MODEL...')
model = CleanViT(
            image_size = args.image_size,
            patch_size = args.patch_size,
            dim = args.dim,
            depth = args.depth,
            heads = args.heads,
            mlp_dim = args.mlp_dim,
            channels = args.channels,
            dim_head = args.dim_head
        )

print('\nNUMBER OF TRAINABLE PARAMETERS:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-07)
ckpt = ModelCheckpoint(dirpath='./checkpoint',
                       filename='./ckpt-{val_psnr:.2f}',
                       monitor='val/loss', 
                       save_last=True)

print('PREPARING LIGHTNING MODULE...')
pl_module = LightningViT(model, optimizer, scheduler)
trainer = Trainer(callbacks=[ckpt], 
                  accelerator=args.device, 
                  max_epochs=args.epochs,
                  logger=logger)

print(f'STARTING TRAINING FOR {args.epochs} EPOCHS')
trainer.fit(pl_module, tr_dl, val_dl, ckpt_path=args.ckpt_path)



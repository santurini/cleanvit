import torch
import torch.nn as nn
from piqa import PSNR, SSIM
import pytorch_lightning as pl
from helpers import JPEGCompressor

class LightningViT(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler, q):
        super().__init__()
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = nn.L1Loss()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.compression = JPEGCompressor(q)
        
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx, stage):
        log_flag = (batch_idx % 30 == 0) & (stage=='train')
        hq = batch; lq = self.compression(batch)
        out = self.forward(lq)
        loss = self.criterion(out, hq)
        psnr = self.psnr(torch.clamp(out, 0, 1), torch.clamp(hq, 0, 1))
        ssim = self.ssim(torch.clamp(out, 0, 1), torch.clamp(hq, 0, 1))
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/psnr', psnr, prog_bar=True)
        self.log(f'{stage}/ssim', ssim, prog_bar=True)
        if log_flag: self.log_images(hq, lq, out)
        outputs = {"loss": loss, "psnr": psnr, "ssim": ssim}
        return outputs
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train") 

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def log_images(self, hq, lq, out):
        gt = [i.unsqueeze(0) for i in hq[:3]]
        pred = [torch.clamp(i, 0, 1).unsqueeze(0) for i in out[:3]]
        inp = [i.unsqueeze(0) for i in lq[:3]]
        cap = ['PSNR: ' + str(self.psnr(i, j).detach().cpu().numpy().round(2)) +
               '\nSSIM: ' + str(self.ssim(i, j).detach().cpu().numpy().round(2)) 
               for i, j in zip(pred, gt)]
        self.logger.log_image(key='Ground Truths', images=gt, caption=['gt']*3)
        self.logger.log_image(key='Predicted Images', images=pred, caption=cap)
        self.logger.log_image(key='Input Images', images=inp, caption=['inp']*3)

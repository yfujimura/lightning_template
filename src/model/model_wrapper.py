import torch
from torch import nn
import torchvision
from einops import rearrange

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from src.model.autoencoder import AutoEncoder

class ModelWrapper(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = AutoEncoder(cfg.model)
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x = rearrange(x, "b c h w -> b (c h w)")
        y = self.model(x)
        loss = self.loss_fn(x, y)
        self.log("train/loss", loss)
        self.log("info/global_step", self.global_step) 
        return loss

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0]
        x = rearrange(x, "b c h w -> b (c h w)")
        y = self.model(x)
        loss = self.loss_fn(x, y)
        self.log("val/loss", loss)

        x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
        input_image = torchvision.transforms.functional.to_pil_image(x[0])
        y = rearrange(y, "b (h w) -> b h w", h=28, w=28)
        pred_image = torchvision.transforms.functional.to_pil_image(y[0].float())
        self.logger.log_image(
            key="samples",
            images=[input_image, pred_image],
            caption=["input", "prediction"],
        )

    def test_step(self, batch, batch_idx):
        x = batch[0]
        x = rearrange(x, "b c h w -> b (c h w)")
        y = self.model(x)
        loss = self.loss_fn(x, y)
        self.log_dict({"test_loss": loss})
        return {'test_loss': loss}

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.optimizer.lr, weight_decay=0.05, betas=(0.9, 0.95))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.trainer.max_steps, eta_min=self.cfg.optimizer.lr * 0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    
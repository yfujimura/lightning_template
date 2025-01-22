import torch
from torchvision import datasets
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule

class DataModule(LightningDataModule):

    def __init__(self, cfg, global_rank=0):
        super().__init__()
        self.cfg = cfg

        self.train_dataset = datasets.MNIST('data', train=True, transform=transforms.ToTensor(), download=True)
        self.val_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor(), download=True)
        self.global_rank = global_rank

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataset.train.batch_size,
            num_workers=self.cfg.dataset.train.num_workers,
            shuffle=True,
            drop_last=False,
        )
        return data_loader

    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataset.val.batch_size,
            num_workers=self.cfg.dataset.val.num_workers,
            shuffle=False,
            drop_last=False,
        )
        return data_loader

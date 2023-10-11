import pytorch_lightning as pl
from core import Config
from torch.utils.data import DataLoader

config = Config()

class DataModuleWrapper(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds=None):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=config.batch_size,
                          shuffle=True,
                          num_workers=config.train_data_workers,
                          pin_memory=True,
                          drop_last=False,
                          )
    
    def val_dataloader(self):
        if self.val_ds is not None:
            return DataLoader(self.val_ds,
                            batch_size=config.test_batch_size,
                            shuffle=False,
                            num_workers=config.test_data_workers,
                            pin_memory=True,
                            drop_last=False,
                            )

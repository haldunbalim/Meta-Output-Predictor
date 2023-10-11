import numpy as np
from torch.utils.data import Dataset
from dyn_models.filtering_lti import *
from core import Config
import torch
import pickle

config = Config()


class FilterDataset(Dataset):
    def __init__(self, path, use_true_len=False):
        super(FilterDataset, self).__init__()
        self.load(path)
        self.use_true_len = use_true_len

    def load(self, path):
        with open(path, "rb") as f:
            self.entries = pickle.load(f)

    def __len__(self):
        return config.train_steps * config.batch_size if not self.use_true_len else len(self.entries)

    def __getitem__(self, idx):
        # generate random entites
        entry = self.entries[idx % len(self.entries)].copy()
        
        if config.dataset_typ in ["ypred", "noniid", "upperTriA"]:
            obs = entry.pop("obs")
            entry["xs"] = obs[:-1]
            entry["ys"] = obs[1:]
        elif config.dataset_typ == "drone":
            obs = entry.pop("obs")
            actions = entry.pop("actions")
            entry["xs"] = np.concatenate([obs[:-1], actions], axis=-1)
            entry["ys"] = obs[1:]
        else:
            raise NotImplementedError(f"{config.dataset_typ} is not implemented")
        
        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in entry.items()])
        return torch_entry

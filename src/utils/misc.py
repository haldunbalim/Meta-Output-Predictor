import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def log_info(st):
    if type(st) != str:
        st = str(st)
    return logger.info(st)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def set_seed(seed=0, fully_reproducible=False):
    # Improve reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if fully_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def plot_errs(names, err_lss, legend_loc="upper right", ax=None, shade=True):
    if ax is None:
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(111)
    ax.set_yscale('log')
    ax.grid()
    handles = []
    for i, (name, err_ls) in enumerate(zip(names, err_lss)):
        traj_errs = err_ls.sum(axis=1)
        print(name, "{:.2f}".format(traj_errs.mean()))
        avg, std = err_ls.mean(axis=0), err_ls.std(axis=0)
        handles.extend(ax.plot(avg, label=name, linewidth=3))
        if shade:
            ax.fill_between(range(len(avg)), avg-std, 
                            avg+std, facecolor=handles[-1].get_color(), alpha=0.33)
        
    ax.legend(fontsize=30, loc=legend_loc)
    ax.set_ylabel("Prediction Error", fontsize=30)
    ax.set_xlabel("t", fontsize=30)
    ax.grid(which="both")
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=20)

def spectrum(A, k):
    spec_rad = np.max(np.abs(np.linalg.eigvals(A)))
    return np.linalg.norm(np.linalg.matrix_power(A,k)) / spec_rad**k

class RLSSingle:
    def __init__(self, ni, lam=1):
        self.lam = lam
        self.P = np.eye(ni)
        self.mu = np.zeros(ni)

    def add_data(self, x, y):
        z = self.P @ x / self.lam
        alpha = 1 / (1 + x.T @ z)
        wp = self.mu + y * z
        self.mu = self.mu + z * (y - alpha * x.T @ wp)
        self.P -= alpha * np.outer(z,z)
    
class RLS:
    
    def __init__(self, ni, no, lam=1):
        self.rlss = [RLSSingle(ni, lam) for _ in range(no)]
        
    def add_data(self, x, y):
        for _y, rls in zip(y, self.rlss):
            rls.add_data(x, _y)
    
    def predict(self, x):
        return np.array([rls.mu @ x for rls in self.rlss])


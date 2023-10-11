import logging
from dyn_models import generate_lti_sample, generate_pendulum_sample, generate_drone_sample
from core import Config
from tqdm import tqdm
import pickle
import os

logger = logging.getLogger(__name__)
config = Config()
config.parse_args()
print("Collecting data for", config.dataset_typ)
samples = []
for name, num_tasks in zip(["train", "val"], [config.num_tasks, config.num_val_tasks]):
    print("Generating", num_tasks, "samples for", name)
    for i in tqdm(range(num_tasks)):
        if config.dataset_typ == "drone":
            sim_obj, sample = generate_drone_sample(
                config.n_positions, sigma_w=1e-1, sigma_v=1e-1,  dt=1e-1)
        else:
            fsim, sample = generate_lti_sample(config.dataset_typ, config.n_positions,
                                              config.nx, config.ny,
                                              sigma_w=1e-1, sigma_v=1e-1, n_noise=config.n_noise)
        samples.append(sample)

    os.makedirs("../data", exist_ok=True)
    with open(f"../data/{name}_{config.dataset_typ}.pkl", "wb") as f:
        pickle.dump(samples, f)

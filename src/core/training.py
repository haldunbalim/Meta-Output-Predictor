import logging
import time
import hashlib
import os
import numpy as np

from core import Config
import os
import glob
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers

# Setup logger
logger = logging.getLogger(__name__)
config = Config()


def setup_train(model):
    output_dir = None
    if config.ckpt_path is not None and config.ckpt_path != '':
        output_dir = "/".join(config.ckpt_path.split("/")[:-2])
        logger.info(f'Resuming from the checkpoint: {config.ckpt_path}')
    if output_dir is None:
        identifier = (model.__class__.__name__ + '/' +
                    time.strftime('%y%m%d_%H%M%S') + '.' +
                    hashlib.md5(config.get_full_yaml().encode('utf-8')).hexdigest()[:6]
                    )

        output_dir = '../outputs/' + identifier

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # Write source code to output dir
        config.write_file_contents(output_dir)

    # Log messages to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(output_dir + '/messages.log')
    file_handler.setFormatter(root_logger.handlers[0].formatter)
    for handler in root_logger.handlers[1:]:  # all except stdout
        root_logger.removeHandler(handler)
    root_logger.addHandler(file_handler)

    # Print model details
    num_params = sum([
        np.prod(p.size())
        for p in filter(lambda p: p.requires_grad, model.parameters())
    ])
    logger.info('\nThere are %d trainable parameters.\n' % num_params)

    return output_dir

def get_callbacks_and_loggers(model, output_dir):
    lr_monitor = pl_callbacks.LearningRateMonitor(logging_interval='epoch')
    tb_logger = pl_loggers.TensorBoardLogger(output_dir)
    loggers = [tb_logger]

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="{step}",
        save_top_k=-1,
        every_n_train_steps=10000,
    )

    ckpt_path = None
    ckpt_paths = glob.glob(os.path.join(output_dir, "checkpoints", "epoch=*"))
    if len(ckpt_paths) > 0:
        ckpt_paths = [int(ckpt_path.split(os.path.sep)[-1]
                        [len("epoch="):len("epoch=")+2]) for ckpt_path in ckpt_paths]
        max_idx = max(ckpt_paths)
        ckpt_path = os.path.join(output_dir, "checkpoints",
                                "epoch={:02d}.ckpt".format(max_idx))
        logger.info("Resuming from checkpoint: {}".format(
            ckpt_path.split(os.path.sep)[-1]))

    callbacks = [checkpoint_callback, lr_monitor]
    return callbacks, loggers

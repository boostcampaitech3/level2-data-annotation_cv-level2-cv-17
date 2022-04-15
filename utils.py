import os
import glob
from pathlib import Path
import re

import random
import numpy as np
import torch


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [
            re.search(rf"%s(\d+)" % path.stem, d) for d in dirs
        ]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        workdir = f"{path}{n}"
        return workdir


def set_seeds(random_seed):
    # for reproducibility
    # os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    np.random.default_rng(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def update_args(args, wandb_cfg):
#     # wandb_cfg changes with every experiment
#     # so, it must be updated to args
#     sweep_cfg = get_sweep_cfg(args)
#     for key in sweep_cfg['parameters'].keys():
#         vars(args)[f"{key}"] = wandb_cfg[f"{key}"]
#     return args


# def get_sweep_cfg(args):
#     sweep_cfg = dict(
#         name=args.sweep_name,   # sweep name
#         method='grid',          # Others : 'bayes', 'random'
#         metric=dict(
#             name='valid/loss',  # anything you are logging on wandb
#             goal='minimize'
#         ),
#         parameters=dict(        # if you want to add new parameter, add here
#             image_size={'values': [1024]},
#             input_size={'values': [512]},
#             batch_size={'values': [16]},
#             learning_rate={'values': [1e-3, 1e-4]},
#             max_epoch={'values': [2]},
#         )
#     )
#     return sweep_cfg

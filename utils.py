import os
import os.path as osp
import glob
from pathlib import Path
import re
import json

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

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

def json_normalize(input_json_path = "../input/data/ICDAR17_Korean/ufo/train.json",
                    output_json_path ="/opt/ml/input/data/ICDAR17_Korean/ufo/train_div.json" ):
    """Function normalize points value to 0-1 for visualization and etc

    Args:
        json_path (str, optional): json path which you want to normalize vertical points . Defaults to "../input/data/ICDAR17_Korean/ufo/train.json".
    """
    
    data = read_json(input_json_path)
    
    for image in data['images']:
        image = data['images'][image]
        img_h = image['img_h']
        img_w = image['img_w']
        for polygon in image['words']:
            for points in image['words'][polygon]['points']:
                points[0] = points[0]/img_w
                points[1] = points[1]/img_h
            image['words'][polygon]['points'] = list(map(lambda x: tuple(x),image['words'][polygon]['points']))

    with open(output_json_path,'w') as f:
        json.dump(data, f, indent=4)

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

def delete_image(json_dir, image_dir, extension):
    json_file = read_json(json_dir)
    
    # delete image with extension file
    del_list = [i for i in json_file['images'].keys() if i.split('.')[-1]==extension]
    for i in del_list:
        os.remove(osp.join(image_dir, i))

def update_json(json_dir, extension):
    json_file = read_json(json_dir)
    
    # train.json save to train_origin.json
    new_json_dir = json_dir[:-5] + '_origin.json'
    with open(new_json_dir, 'w') as f:
        json.dump(json_file, f, indent=4)

    # pop extension file
    update_list = [i for i in json_file['images'].keys() if i.split('.')[-1]==extension]
    for i in update_list:
        json_file['images'].pop(i)
    
    # update train.json without extension file
    with open(json_dir, 'w') as f:
        json.dump(json_file, f, indent=4)
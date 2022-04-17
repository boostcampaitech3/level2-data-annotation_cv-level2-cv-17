import os
import os.path as osp

import json

from utils import read_json


def main(load_json_dir = '/opt/ml/input/data/ICDAR17_ALL/ufo/train.json',
         save_json_dir = '/opt/ml/input/data/ICDAR17_ALL/ufo/train_delgif.json',
         image_dir = '/opt/ml/input/data/ICDAR17_ALL/images'):
    
    json_file = read_json(load_json_dir)

    gif_list = [i for i in json_file['images'].keys() if i.split('.')[-1]=='gif']
    for i in gif_list:
        json_file['images'].pop(i)
        
        os.remove(osp.join(image_dir, i))

    with open(save_json_dir, 'w') as f:
        json.dump(json_file, f, indent=4)


if __name__ == '__main__':
    main()
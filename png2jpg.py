import os
import os.path as osp

import json
import copy
from PIL import Image

from utils import read_json


def main(load_json_dir = '/opt/ml/input/data/ICDAR17_ALL/ufo/train_delgif.json',
         image_dir = '/opt/ml/input/data/ICDAR17_ALL/images'):

    json_file = read_json(load_json_dir)

    png_list = [i for i in json_file['images'].keys() if i.split('.')[-1]=='png']
    for png_file in png_list:
        jpg_file = png_file.split('.')[0]+'.jpg'

        json_file['images'][str(jpg_file)] = copy.deepcopy(json_file['images'][str(png_file)])
        json_file['images'].pop(png_file)
        
        img = Image.open(osp.join(image_dir, png_file))
        rgb_img = img.convert('RGB')
        rgb_img.save(osp.join(image_dir, jpg_file))


if __name__ == '__main__':
    main()
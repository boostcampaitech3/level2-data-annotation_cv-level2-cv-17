import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from imageio import imread
from detect import detect
from utils import json_normalize

CHECKPOINT_EXTENSIONS = ['.pth', '.ckpt']


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--model_dir', default=os.environ.get('SM_CHANNEL_MODEL', 'trained_models'))
    parser.add_argument('--output_dir', default=os.environ.get('SM_OUTPUT_DATA_DIR', 'predictions'))

    parser.add_argument('--inf_viz', default=False)

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=20)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_inference(model, ckpt_fpath, data_dir, input_size, batch_size, split='public', inf_viz=False):
    if ckpt_fpath is not None:
        model.load_state_dict(torch.load(ckpt_fpath, map_location='cpu'))
    model.eval()

    image_fnames, by_sample_bboxes = [], []

    images = []
    for image_fpath in tqdm(sorted(glob(osp.join(data_dir, '{}/*'.format(split))))):
        image_fnames.append(osp.basename(image_fpath))

        images.append(cv2.imread(image_fpath)[:, :, ::-1])
        if len(images) == batch_size:
            by_sample_bboxes.extend(detect(model, images, input_size))
            images = []

    if len(images):
        by_sample_bboxes.extend(detect(model, images, input_size))

    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        if inf_viz: 
            # add illegibility for dataviz.py
            words_info = {idx: dict(points=bbox.tolist(), illegibility=False) for idx, bbox in enumerate(bboxes)}
            # add tags for dataviz.py and add img_w, img_h for points normalize(json_normalize)
            image = imread(osp.join(data_dir, 'images', image_fname))
            ufo_result['images'][image_fname] = dict(words=words_info, tags=None, img_w=image.shape[1], img_h=image.shape[0])
        else:
            words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
            ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result


def main(args):
    # Initialize model
    model = EAST(pretrained=False).to(args.device)

    # Get paths to checkpoint files
    ckpt_fpath = osp.join(args.model_dir, 'latest.pth')

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Inference in progress')

    ufo_result = dict(images=dict())
    if args.inf_viz:
        result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                              args.batch_size, split='images', inf_viz=args.inf_viz)
        ufo_result['images'].update(result['images'])
        output_fname = 'output.json'
    else:
        for split in ['public', 'private']:
            print('Split: {}'.format(split))
            split_result = do_inference(model, ckpt_fpath, args.data_dir, args.input_size,
                                        args.batch_size, split=split)
            ufo_result['images'].update(split_result['images'])
        output_fname = 'output.csv'

    with open(osp.join(args.output_dir, output_fname), 'w') as f:
        json.dump(ufo_result, f, indent=4)

    if args.inf_viz:
        json_normalize(input_json_path=osp.join(args.output_dir, output_fname),
                       output_json_path=osp.join(args.output_dir, 'normalize_output.json'))

if __name__ == '__main__':
    args = parse_args()
    main(args)

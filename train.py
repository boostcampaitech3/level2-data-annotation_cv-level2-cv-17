import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import gc
import yaml
import wandb
from functools import reduce, partial

from sweep import update_args, get_sweep_cfg
from utils import increment_path, set_seeds


def parse_args():
    parser = ArgumentParser()
    # directory
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--val_data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--work_dir', type=str, default='./work_dirs',
                        help='the root dir to save logs and models about each experiment')
    # run environment
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=5)
    # training parameter
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    # etc
    parser.add_argument('--sweep', type=bool, default=True, help='sweep option')

    args = parser.parse_args()
    if args.input_size % 32 != 0: raise ValueError('`input_size` must be a multiple of 32')
    return args


def do_training(
    data_dir, val_data_dir, work_dir, work_dir_exp,
    device, seed, num_workers, save_interval,
    image_size, input_size, batch_size, learning_rate, max_epoch,
    sweep
    ):
    set_seeds(seed)
    
    dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # train_copy.json is just copying train.json
    val_dataset = SceneTextDataset(val_data_dir, split='train_copy', image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    
    for epoch in range(max_epoch):
        # train
        model.train()
        epoch_loss, epoch_cls_loss, epoch_ang_loss, epoch_iou_loss = 0, 0, 0, 0
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {} Train]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                epoch_cls_loss += extra_info['cls_loss']
                epoch_ang_loss += extra_info['angle_loss']
                epoch_iou_loss += extra_info['iou_loss']
                
                pbar.update(1)
            # now, set_postfix is about one epoch
            pbar.set_postfix({
                'loss': epoch_loss/num_batches, 'cls_loss': epoch_cls_loss/num_batches,
                'ang_loss': epoch_ang_loss/num_batches, 'iou_loss': epoch_iou_loss/num_batches
            })
        # valid
        model.eval()
        val_epoch_loss, val_epoch_cls_loss, val_epoch_ang_loss, val_epoch_iou_loss = 0, 0, 0, 0
        with tqdm(total=val_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                pbar.set_description('[Epoch {} Valid]'.format(epoch + 1))

                with torch.no_grad():
                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)

                loss_value = loss.item()
                val_epoch_loss += loss_value
                val_epoch_cls_loss += extra_info['cls_loss']
                val_epoch_ang_loss += extra_info['angle_loss']
                val_epoch_iou_loss += extra_info['iou_loss']

                pbar.update(1)
            # now, set_postfix is about one epoch
            pbar.set_postfix({
                'loss': val_epoch_loss/val_num_batches, 'cls_loss': val_epoch_cls_loss/val_num_batches,
                'ang_loss': val_epoch_ang_loss/val_num_batches, 'iou_loss': val_epoch_iou_loss/val_num_batches
            })

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(work_dir_exp, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        # "train/" and "valid/" : the expression for distinction in wandb chart
        wandb.log({
            "train/loss": epoch_loss/num_batches, "valid/loss": val_epoch_loss/val_num_batches,
            "train/cls_loss": epoch_cls_loss/num_batches, "valid/cls_loss": val_epoch_cls_loss/val_num_batches,
            "train/ang_loss": epoch_ang_loss/num_batches, "valid/ang_loss": val_epoch_ang_loss/val_num_batches,
            "train/iou_loss": epoch_iou_loss/num_batches, "valid/iou_loss": val_epoch_iou_loss/val_num_batches
        })


def main(args):
    # generate work directory every experiment
    args.work_dir_exp = increment_path(osp.join(args.work_dir, 'exp'))
    if not osp.exists(args.work_dir_exp): os.makedirs(args.work_dir_exp)
    
    if args.sweep:
        # if you want to use tags, put tags=['something'] in wandb.init
        wandb_run = wandb.init(config=args.__dict__, reinit=True)
        wandb_run.name = args.work_dir_exp.split('/')[-1]  # run name
        args = update_args(args, wandb.config)
        do_training(**args.__dict__)
        wandb_run.finish()
    else:
        # you must to change project name
        # if you want to use tags, put tags=['something'] in wandb.init
        # if you want to use group, put group='something' in wandb.init
        wandb.init(
            entity='mg_generation', project='data_annotation_dongwoo',
            name=args.work_dir_exp.split('/')[-1],
            config=args.__dict__, reinit=True
        )
        do_training(**args.__dict__)

    # save args as yaml file every experiment
    yamldir = osp.join(os.getcwd(), args.work_dir_exp+'/train_config.yml')
    with open(yamldir, 'w') as f: yaml.dump(args.__dict__, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    if args.sweep:
        sweep_cfg = get_sweep_cfg()
        # you must to change project name
        sweep_id = wandb.sweep(sweep=sweep_cfg, entity='mg_generation', project='data_annotation_dongwoo')
        wandb.agent(sweep_id=sweep_id, function=partial(main, args))
    else:
        main(args)

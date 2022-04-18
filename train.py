import os
import os.path as osp
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import lr_scheduler

import math
import json
import yaml
from glob import glob
from tqdm import tqdm
from functools import reduce, partial
import wandb

# baseline
from dataset import SceneTextDataset
from east_dataset import EASTDataset
from model import EAST
from inference import do_inference
from deteval import calc_deteval_metrics

# ours
from sweep import update_args, get_sweep_cfg
from utils import increment_path, set_seeds, read_json
from custom_scheduler import CosineAnnealingWarmUpRestarts


def parse_args():
    parser = ArgumentParser()
    # directory
    parser.add_argument('--data_dir', type=str, nargs='+', default=['/opt/ml/input/data/ICDAR17_ALL','/opt/ml/input/data/ICDAR17_Korean'],
                        help='the dir that have images and ufo/train.json in sub_directories')
    parser.add_argument('--val_data_dir', type=str, default='/opt/ml/input/data/AIHUB_outside_sample',
                        help='the dir that have images and ufo/valid.json in sub_directories')
    parser.add_argument('--work_dir', type=str, default='./work_dirs',
                        help='the root dir to save logs and models about each experiment')
    # run environment
    parser.add_argument('--device', type=str, default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader num_workers')
    parser.add_argument('--save_interval', type=int, default=5, help='model save interval')
    parser.add_argument('--save_max_num', type=int, default=10, help='the max number of model save files')
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluation metric log interval')
    # training parameter
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--optm', type=str, default='adam')
    parser.add_argument('--schd', type=str, default='multisteplr')
    # etc
    parser.add_argument('--sweep', type=bool, default=False, help='sweep option')

    args = parser.parse_args()
    if args.input_size % 32 != 0: raise ValueError('`input_size` must be a multiple of 32')
    return args


def do_training(
    data_dir, val_data_dir, work_dir, work_dir_exp,
    device, seed, num_workers, save_interval, save_max_num, eval_interval,
    image_size, input_size, batch_size, learning_rate, max_epoch, optm, schd,
    sweep
    ):
    set_seeds(seed)

    # train CV dataset
    dataset = [SceneTextDataset(i, split='train', image_size=image_size, crop_size=input_size) for ind, i in enumerate(data_dir)]
    dataset = EASTDataset(ConcatDataset(dataset))
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # valid CV dataset
    val_dataset = SceneTextDataset(val_data_dir, split='valid', image_size=image_size, crop_size=input_size)
    val_dataset = EASTDataset(val_dataset)
    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    # optimizer
    # if you want to use CosineAnnealingWarmUpRestarts, optimizer must be started at lr=0
    if optm == 'adam':
        if schd == 'cosignlr':
            optimizer = torch.optim.Adam(model.parameters(), lr=0)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optm == 'sgd':
        if schd == 'cosignlr':
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # scheduler
    if schd == 'multisteplr':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
    elif schd == 'reducelr':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    elif schd == 'cosignlr':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=max_epoch, T_mult=1, eta_max=learning_rate, T_up=max_epoch//10, gamma=0.5)
    
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
            # set_postfix is about one epoch
            pbar.set_postfix({
                'loss': epoch_loss/num_batches, 'cls_loss': epoch_cls_loss/num_batches,
                'ang_loss': epoch_ang_loss/num_batches, 'iou_loss': epoch_iou_loss/num_batches,
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
            # set_postfix is about one epoch
            pbar.set_postfix({
                'loss': val_epoch_loss/val_num_batches, 'cls_loss': val_epoch_cls_loss/val_num_batches,
                'ang_loss': val_epoch_ang_loss/val_num_batches, 'iou_loss': val_epoch_iou_loss/val_num_batches,
            })

        # valid evaluation
        if (epoch + 1) % eval_interval == 0:
            gt_ufo = read_json(osp.join(val_data_dir, 'ufo/valid.json'))
            # ckpt_fpath : we don't use in here
            # split : valid image_folder_name
            pred_ufo = do_inference(model=model, input_size=input_size, batch_size=batch_size,
                                    data_dir=val_data_dir, ckpt_fpath=None, split='images')
            val_epoch_precison, val_epoch_recall, val_epoch_hmean = do_evaluating(gt_ufo, pred_ufo)
            wandb.log({
                "valid_metric/precision": val_epoch_precison,
                "valid_metric/recall": val_epoch_recall,
                "valid_metric/hmean": val_epoch_hmean,
            }, commit=False)

        # ReduceLROnPlateau scheduler consider valid loss when doing step
        if schd == 'reducelr':
            scheduler.step(val_epoch_loss)
        else:
            scheduler.step()
        
        # save model checkpoint
        if (epoch + 1) % save_interval == 0:
            ckpt_fpath = osp.join(work_dir_exp, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            pth_files = glob(osp.join(work_dir_exp,'*.pth'))
            if len(pth_files) > save_max_num:
                epoch_num_list = [f.split('/')[-1].split('.')[0].split('_')[-1] for f in pth_files]
                min_epoch_num = sorted(epoch_num_list, key=lambda x: int(x))[0]
                os.remove(osp.join(work_dir_exp, f'epoch_{min_epoch_num}.pth'))

        wandb.log({
            "train/loss": epoch_loss/num_batches, "valid/loss": val_epoch_loss/val_num_batches,
            "train/cls_loss": epoch_cls_loss/num_batches, "valid/cls_loss": val_epoch_cls_loss/val_num_batches,
            "train/ang_loss": epoch_ang_loss/num_batches, "valid/ang_loss": val_epoch_ang_loss/val_num_batches,
            "train/iou_loss": epoch_iou_loss/num_batches, "valid/iou_loss": val_epoch_iou_loss/val_num_batches,
        }, commit=True)  # commit=True : It notify that one epoch is ended with this log.  # default=True


def do_evaluating(gt_ufo, pred_ufo):
    epoch_precison, epoch_recall, epoch_hmean = 0, 0, 0
    num_images = len(gt_ufo['images'])
    
    for pred_image, gt_image in zip(sorted(pred_ufo['images'].items()), sorted(gt_ufo['images'].items())):
        pred_bboxes_dict, gt_bboxes_dict, gt_trans_dict = {}, {}, {}
        pred_bboxes_list, gt_bboxes_list, gt_trans_list = [], [], []
        
        for pred_point in range(len(pred_image[1]['words'])):
            pred_bboxes_list.extend([pred_image[1]['words'][pred_point]['points']])
        pred_bboxes_dict[pred_image[0]] = pred_bboxes_list
        
        for gt_point in range(len(gt_image[1]['words'])):
            gt_bboxes_list.extend([gt_image[1]['words'][str(gt_point)]['points']])
            gt_trans_list.extend([gt_image[1]['words'][str(gt_point)]['transcription']])
        gt_bboxes_dict[gt_image[0]] = gt_bboxes_list
        gt_trans_dict[gt_image[0]] = gt_trans_list
        
        # eval_metric['total'] : this value is about all of bboxes in one image
        eval_metric = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict=gt_trans_dict)
        epoch_precison += eval_metric['total']['precision']
        epoch_recall += eval_metric['total']['recall']
        epoch_hmean += eval_metric['total']['hmean']
    
    return epoch_precison/num_images, epoch_recall/num_images, epoch_hmean/num_images


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

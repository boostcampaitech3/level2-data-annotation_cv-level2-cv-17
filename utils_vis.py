import os.path as osp
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import lanms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric.resize import LongestMaxSize
import torch
from dataset import get_rotate_mat

def draw_bbox(image, bbox, color=(0,0,255), thickness=1, thickness_sub=None, double_lined=False, write_point_numbers=True):
    '''
        image : (H, W, C)
        bbox : (n, 4, 2)
    '''
    thickness_sub = thickness_sub or thickness*3
    basis = max(image.shape[:2]) # (H, W, C) 니까  max(H, W)
    fontsize = basis / 1500
    x_offset, y_offset = int(fontsize * 12), int(fontsize * 10)
    color_sub = (255 - color[0], 255 - color[1], 255 - color[2])
    points = [(int(np.rint(p[0])), int(np.rint(p[1]))) for p in bbox]

    for idx in range(len(points)):
        if double_lined:
            cv2.line(image, points[idx], points[(idx + 1) % len(points)], color_sub,
                     thickness=thickness_sub)
        cv2.line(image, points[idx], points[(idx + 1) % len(points)], color, thickness=thickness)

    if write_point_numbers:
        for idx in range(len(points)):
            loc = (points[idx][0] - x_offset, points[idx][1] - y_offset)
            if double_lined:
                cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color_sub,
                            thickness_sub, cv2.LINE_AA)
            cv2.putText(image, str(idx), loc, cv2.FONT_HERSHEY_SIMPLEX, fontsize, color, thickness,
                        cv2.LINE_AA)

def draw_bboxes(image, bboxes, color=(0, 0, 255), thickness=1, thickness_sub=None,
                double_lined=False, write_point_numbers=True):
    for bbox in bboxes:
        draw_bbox(image, bbox, color=color, thickness=thickness, thickness_sub=thickness_sub,
                  double_lined=double_lined, write_point_numbers=write_point_numbers)

def gray_mask_to_heatmap(x):
    x = cv2.cvtColor(cv2.applyColorMap(x, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return x

def get_superimposed_image(image, score_map, heatmap=True, w_image=None, w_map=None):
    """
    Args:
        image (ndarray): (H, W, C) shaped, float32 or uint8 dtype is allowed.
        score_map (ndarray): (H, W) shaped, float32 or uint8 dtype is allowed.
        heatmap (boot): Wheather to convert `score_map` into a heatmap.
        w_image (float)
        w_map (float)

    Blending weights(`w_image` and `w_map`) are default to (0.4, 0.6).
    """

    assert w_image is None or (w_image > 0 and w_image < 1)
    assert w_map is None or (w_map > 0 and w_map < 1)

    if image.dtype != np.uint8:
        image = (255 * np.clip(image, 0, 1)).astype(np.uint8)

    if score_map.dtype != np.uint8:
        score_map = (255 * np.clip(score_map, 0, 1)).astype(np.uint8)
    if heatmap:
        score_map = gray_mask_to_heatmap(score_map)
    elif score_map.ndim == 2 or score_map.shape[2] != 3:
        score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2RGB)

    if w_image is None and w_map is None:
        w_image, w_map = 0.4, 0.6
    elif w_image is None:
        w_image = 1 - w_map
    elif w_map is None:
        w_map = 1 - w_image

    return cv2.addWeighted(image, w_image, score_map, w_map, 0)

def find_bbox_from_maps(score_map, geo_map, orig_size, input_size, score_thresh, nms_thresh):
    xy_text = np.argwhere(score_map > score_thresh)[:, ::-1].copy()  # (n x 2)
    if xy_text.size == 0:
        bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        return bboxes
    else:
        xy_text = xy_text[np.argsort(xy_text[:, 1])]  # Row-wise로 정렬
        valid_pos = xy_text * int(1 / 0.25)
        valid_geo = geo_map[xy_text[:, 1], xy_text[:, 0], :]  # (n x 5)
        indices, bboxes = [], []
        for idx, ((x, y), g) in enumerate(zip(valid_pos, valid_geo)):
            y_min, y_max = y - g[0], y + g[1]
            x_min, x_max = x - g[2], x + g[3]
            rotate_mat = get_rotate_mat(-g[4])

            bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            anchor = np.array([x, y], dtype=np.float32).reshape(2, 1)
            rotated_bbox = (np.dot(rotate_mat, bbox.T - anchor) + anchor).T

            if bbox[:, 0].min() < 0 or bbox[:, 0].max() >= score_map.shape[1] * int(1 / 0.25):
                continue
            elif bbox[:, 1].min() < 0 or bbox[:, 1].max() >= score_map.shape[0] * int(1 / 0.25):
                continue

            indices.append(idx)
            bboxes.append(rotated_bbox.flatten())
        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
            return bboxes
        bboxes = np.array(bboxes)
        # 좌표 정보에 Score map에서 가져온 Score를 추가
        scored_bboxes = np.zeros((bboxes.shape[0], 9), dtype=np.float32)
        scored_bboxes[:, :8] = bboxes
        scored_bboxes[:, 8] = score_map[xy_text[indices, 1], xy_text[indices, 0]]

        # LA-NMS 적용
        nms_bboxes = lanms.merge_quadrangle_n9(scored_bboxes.astype('float32'), nms_thresh)
        nms_bboxes = nms_bboxes[:, :8].reshape(-1, 4, 2)
        
        # 원본 이미지 크기에 맞게 bbox 크기 보정
        nms_bboxes *= max(orig_size) / input_size
        return nms_bboxes

def convert_to_bbox(score_map, geo_map):
    xy_text = np.argwhere(score_map > 0.9)[:, ::-1].copy()
    if xy_text.size == 0:
        bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        return bboxes
    else:
        xy_text = xy_text[np.argsort(xy_text[:, 1])]  # Row-wise로 정렬
        valid_pos = xy_text * 4
        valid_geo = geo_map[xy_text[:, 1], xy_text[:, 0], :]  # (n x 5)
        indices, bboxes = [], []
        for idx, ((x, y), g) in enumerate(zip(valid_pos, valid_geo)):
            y_min, y_max = y - g[0], y + g[1]
            x_min, x_max = x - g[2], x + g[3]
            rotate_mat = get_rotate_mat(-g[4])
            bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            anchor = np.array([x, y], dtype=np.float32).reshape(2, 1)
            rotated_bbox = (np.dot(rotate_mat, bbox.T - anchor) + anchor).T
            if bbox[:, 0].min() < 0 or bbox[:, 0].max() >= score_map.shape[1] * 4:
                continue
            elif bbox[:, 1].min() < 0 or bbox[:, 1].max() >= score_map.shape[0] * 4:
                continue
            indices.append(idx)
            bboxes.append(rotated_bbox.flatten())
        if len(bboxes) == 0:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
            return bboxes
        bboxes = np.array(bboxes)
        scored_bboxes = np.zeros((bboxes.shape[0], 9), dtype=np.float32)
        scored_bboxes[:, :8] = bboxes
        scored_bboxes[:, 8] = score_map[xy_text[indices, 1], xy_text[indices, 0]]
        nms_bboxes = lanms.merge_quadrangle_n9(scored_bboxes.astype('float32'), 0.2)
        nms_bboxes = nms_bboxes[:, :8].reshape(-1, 4, 2)
        return nms_bboxes
        


def detect_valid(image, score_map, geo_map, score_thresh=0.9, figsize=(8, 8), nms_thresh = 0.2):
    score_map, geo_map = score_map.permute(1, 2, 0).cpu().numpy(), geo_map.permute(1, 2, 0).cpu().numpy()
    score_map = score_map.squeeze()
    center_mask = score_map > score_thresh
    fig, axs = plt.subplots(ncols=6, nrows=1, figsize=(figsize[0] * 6, figsize[1]))
    axs[0].imshow(center_mask, cmap='gray')
    for idx in range(4):
        axs[idx + 1].imshow(geo_map[:, :, idx] * center_mask, cmap='jet')
    axs[5].imshow(geo_map[:, :, 4] * center_mask)
    bboxes = convert_to_bbox(score_map, geo_map)
    vis = image.permute(1,2,0).cpu().numpy().copy()
    vis = vis * (np.array([0.5,0.5,0.5], dtype=np.float32) * 255.0) + (np.array([0.5,0.5,0.5], dtype=np.float32) * 255.0)
    draw_bboxes(vis, bboxes, thickness=2)
    return fig, vis


# def detect_valid(model, images, input_size, score_thresh=0.9, figsize=(8, 8), nms_thresh = 0.2):
#     '''
#         image : numpy image(H, W, C)
#     '''
#     prep_fn = A.Compose([
#         LongestMaxSize(input_size), A.PadIfNeeded(min_height=input_size, min_width=input_size,
#                                                   position=A.PadIfNeeded.PositionType.TOP_LEFT),
#         A.Normalize(), ToTensorV2()])
#     device = list(model.parameters())[0].device

#     # batch 개수만큼 이미지 리스트 저장
#     batch, orig_sizes = [], []
#     for image in images:
#         orig_sizes.append(image.shape[:2])
#         batch.append(prep_fn(image=image)['image'])
#     batch = torch.stack(batch, dim=0).to(device) # b, 3, 512, 512

#     with torch.no_grad():
#         score_maps, geo_maps = model(batch)
#     score_maps, geo_maps = score_maps.cpu().numpy(), geo_maps.cpu().numpy()
#     by_sample_bboxes = []
#     by_sample_maps = []
#     for img, score_map, geo_map, orig_size in zip(batch, score_maps, geo_maps, orig_sizes):
#         score_map, geo_map = score_map.transpose(1, 2, 0), geo_map.transpose(1, 2, 0)
#         score_map = score_map.squeeze()

#         # draw map
#         map_margin = int(abs(orig_size[0] - orig_size[1]) * 0.25 * input_size / max(orig_size))
#         print(f'map_margin : {map_margin}')
#         if map_margin > 0:
#             if orig_size[0] > orig_size[1]:
#                 score_map[:, -map_margin:] = 0
#                 geo_map[:, -map_margin:, :] = 0
#             else:
#                 score_map[-map_margin:, :] = 0
#                 geo_map[-map_margin:, :, :] = 0

#         center_mask = score_map > score_thresh
#         fig, axs = plt.subplots(ncols=6, nrows=1, figsize=(figsize[0] * 6, figsize[1]))
#         axs[0].imshow(center_mask, cmap='gray')
#         for idx in range(4):
#             axs[idx + 1].imshow(geo_map[:, :, idx] * center_mask, cmap='jet')
#         axs[5].imshow(geo_map[:, :, 4] * center_mask)
#         by_sample_maps.append(fig)

#         # bbox
#         bboxes = find_bbox_from_maps(score_map, geo_map, orig_size, input_size, score_thresh, nms_thresh)
#         d_img = img.permute(1,2,0).detach().cpu().numpy()
#         vis = d_img.copy()
#         draw_bboxes(vis, bboxes, thickness=2)
#         by_sample_bboxes.append(vis)
#         # bboxes = draw_bbox(score_map, geo_map)
#         # if bboxes is None:
#         #     bboxes = np.zeros((0, 4, 2), dtype=np.float32)
#         # else:
#         #     bboxes = bboxes[:, :8].reshape(-1, 4, 2)
#         #     bboxes *= max(orig_size) / input_size
#         # by_sample_bboxes.append(bboxes)

#     return by_sample_bboxes, by_sample_maps
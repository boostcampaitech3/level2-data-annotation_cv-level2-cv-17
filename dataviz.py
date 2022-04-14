import glob
import fiftyone as fo
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from PIL import Image
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont


images_path = "/opt/ml/input/data/ICDAR17_Korean/images/*"# "/path/to/images/*"

# Ex: your custom label format
# annotations = {
#     "/path/to/images/000001.jpg": [
#         {"bbox": ..., "label": ...},
#         ...
#     ],
#     ...
# }
def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

annotations = read_json("../input/data/ICDAR17_Korean/ufo/train.json")

# {'img_4380.jpg': {'img_h': 2448,
#    'img_w': 1836,
#    'words': {'0': {'points': [[662.0, 747.0], # 이게 top left임. 이거에다가, width, height만 넣으면 되는 듯?
#       [945.0, 759.0],
#       [922.0, 1582.0],
#       [673.0, 1565.0]],
#      'transcription': '출입금지',
#      'language': ['ko'],
#      'illegibility': False,
#      'orientation': 'Horizontal',
#      'word_tags': None},
#     '1': {'points': [[476.0, 551.0],
#       [1132.0, 554.0],
#       [1118.0, 747.0],
#       [471.0, 716.0]],
#      'transcription': '오토바이',
#      'language': ['ko'],
#      'illegibility': False,
#      'orientation': 'Horizontal',
#      'word_tags': None},
#     '2': {'points': [[455.0, 293.0],
#       [1144.0, 310.0],
#       [1129.0, 518.0],
#       [457.0, 496.0]],
#      'transcription': '자전거',

# Create samples for your data
samples = []
for filepath in glob.glob(images_path):
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath.split('/')[-1]]:
        img_h = obj["img_h"]
        img_w = obj["img_w"]

        # transcription



        # words를 하나씩 돌면서 이미지 크기로 나눠줘야겠다.
        for word in obj['words']:
            x1 = word['points'][0][0]/img_w
            y1 = word['points'][0][1]/img_h
            bbox = [x1,y1,img_w,img_h]
            # tag = word['tag']



        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        # bounding_box = obj["bbox"]

        detections.append(
            fo.Detection( bounding_box=bbox)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    samples.append(sample)

# Create dataset
dataset = fo.Dataset("my-detection-dataset")
dataset.add_samples(samples)
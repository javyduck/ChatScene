import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
LAB_FORMATS = 'txt'  # include label suffixes
NPY_FORMATS = 'npy'
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

class LoadImagesAndBoxLabels(Dataset):
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, 'images/*.*'))))  # dir
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni = len(images)

        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, 'labels/*.*'))))  # dir
            else:
                raise FileNotFoundError(f'{p} does not exist')

        labels = [x for x in files if x.split('.')[-1].lower() in LAB_FORMATS]
        self.labels = []
        for lb in labels:
            lb_arr = np.loadtxt(lb, delimiter=' ')
            if len(lb_arr.shape) == 1:
                lb_arr = np.expand_dims(lb_arr, 0)
            self.labels.append(lb_arr[[0], :])
        self.labels = np.array(self.labels)
        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.files_label = labels
        self.nf = ni   # number of files
        self.mode = 'image'
        self.auto = auto

        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __getitem__(self, index):
        # print(self.files[index])
        img0 = cv2.imread(self.files[index])
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        ret_label = np.loadtxt(self.files_label[index], delimiter=' ')
        if len(ret_label.shape) > 1:
            for idx in range(ret_label.shape[0]):
                if ret_label[idx, 0] == 11:
                    ret_label = ret_label[idx]
                    break
        
        img = torch.from_numpy(img) / 255.
        img0 = torch.from_numpy(img0) / 255.
        ret_label = torch.from_numpy(ret_label)
        return img, img0, ret_label

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

    @staticmethod
    def collate_fn(batch):
        img, img0, ret_label= zip(*batch)  # transposed
        img_id = torch.arange(0, len(batch), 1).unsqueeze(-1)

        return torch.stack(img, 0), torch.stack(img0, 0), torch.cat([img_id, torch.stack(ret_label, 0)], dim=-1)
        
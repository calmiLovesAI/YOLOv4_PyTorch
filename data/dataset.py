import torch
import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from configuration import Config


class YoloDataset(Dataset):
    def __init__(self):
        self.annotation_dir = Config.txt_file_dir

    def __len__(self):
        with open(file=self.annotation_dir, mode="r") as f:
            annotation_list = f.readlines()
        return len(annotation_list)

    def __getitem__(self, item):
        pass
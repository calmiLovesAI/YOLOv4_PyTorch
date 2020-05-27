import torch
import numpy as np
import cv2
import math
import random

from torch.utils.data import Dataset
from configuration import Config


class YoloDataset(Dataset):
    def __init__(self, shuffle=True, transform=None):
        self.annotation_dir = Config.txt_file_dir
        self.batch_size = Config.batch_size
        self.max_boxes_per_image = Config.max_boxes_per_image
        self.transform = transform

        with open(file=self.annotation_dir, mode="r") as f:
            self.annotation_list = f.readlines()

        if shuffle:
            random.shuffle(self.annotation_list)

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, item):
        annotation = self.annotation_list[item]
        image_array, labels_array = self.__get_image_information(line_string=annotation)
        sample = {
            "image": image_array,
            "label": labels_array
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __get_image_information(self, line_string):
        """
        :param line_string: string
        :return:
        image_file: numpy array, shape = (H, W, C)
        labels_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
        """
        line_list = line_string.strip().split(" ")
        image_file_dir, image_height, image_width = line_list[:3]
        image_array = cv2.imread(filename=image_file_dir)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_height, image_width = float(image_height), float(image_width)
        boxes = []
        num_of_boxes = (len(line_list) - 3) / 5
        if int(num_of_boxes) == num_of_boxes:
            num_of_boxes = int(num_of_boxes)
        else:
            raise ValueError("num_of_boxes must be type 'int'.")
        for index in range(num_of_boxes):
            if index < self.max_boxes_per_image:
                xmin = int(float(line_list[3 + index * 5]))
                ymin = int(float(line_list[3 + index * 5 + 1]))
                xmax = int(float(line_list[3 + index * 5 + 2]))
                ymax = int(float(line_list[3 + index * 5 + 3]))
                class_id = int(line_list[3 + index * 5 + 4])
                boxes.append([xmin, ymin, xmax, ymax, class_id])
        num_padding_boxes = self.max_boxes_per_image - num_of_boxes
        if num_padding_boxes > 0:
            for i in range(num_padding_boxes):
                boxes.append([0, 0, 0, 0, -1])
        labels_array = np.array(boxes, dtype=np.float32)
        return image_array, labels_array


class BatchDataset:
    def __init__(self, dataset, batch_size, drop_remainder=False):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder

        quotient = self.dataset_size / self.batch_size
        remainder = self.dataset_size % self.batch_size
        self.steps_per_epoch = math.floor(quotient)
        if remainder and not self.drop_remainder:
            self.steps_per_epoch = math.ceil(quotient)

    def __len__(self):
        return self.steps_per_epoch

    def batch(self, index):
        images_list = []
        labels_list = []
        for i in range(self.batch_size):
            dataset_index = index * self.batch_size + i
            if dataset_index in range(self.dataset_size):
                images_list.append(self.dataset[dataset_index]["image"])
                labels_list.append(self.dataset[dataset_index]["label"])
            else:
                break

        batch_images = torch.stack(tensors=images_list, dim=0)
        batch_labels = torch.stack(tensors=labels_list, dim=0)
        return {
            "images": batch_images,
            "labels": batch_labels
        }

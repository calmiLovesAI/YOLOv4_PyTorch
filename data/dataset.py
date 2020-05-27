import numpy as np
import cv2
import math
import random

from torch.utils.data import Dataset
from configuration import Config


class YoloDataset(Dataset):
    def __init__(self, drop_remainder=True, shuffle=True, transform=None):
        self.annotation_dir = Config.txt_file_dir
        self.batch_size = Config.batch_size
        self.max_boxes_per_image = Config.max_boxes_per_image
        self.transform = transform

        with open(file=self.annotation_dir, mode="r") as f:
            self.annotation_list = f.readlines()

        quotient = len(self.annotation_list) / self.batch_size
        remainder = len(self.annotation_list) % self.batch_size
        self.steps_per_epoch = math.floor(quotient)
        if remainder and not drop_remainder:
            self.steps_per_epoch = math.ceil(quotient)

        if shuffle:
            random.shuffle(self.annotation_list)

    def __len__(self):
        return len(self.annotation_list)

    def __getitem__(self, item):
        annotations = []
        if item < self.steps_per_epoch:
            annotations = self.annotation_list[item * self.batch_size: (item + 1) * self.batch_size]
        elif item == self.steps_per_epoch:
            annotations = self.annotation_list[item * self.batch_size:]
        image_list = []
        labels_list = []
        image_height_list = []
        image_width_list = []
        for i in range(len(annotations)):
            image_array, labels_array, image_height, image_width = self.__get_image_information(line_string=annotations[i])
            image_list.append(image_array)
            labels_list.append(labels_array)
            image_height_list.append(image_height)
            image_width_list.append(image_width)
        images = np.stack(image_list, axis=0)
        labels = np.stack(labels_list, axis=0)
        image_heights = np.stack(image_height_list, axis=0)
        image_widths = np.stack(image_width_list, axis=0)
        sample = {
            "images": images,
            "labels": labels,
            "image_heights": image_heights,
            "image_widths": image_widths
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __get_image_information(self, line_string):
        """
        :param line_string: string
        :return:
        image_file: numpy array, shape = (H, W, C)
        boxes_array: numpy array, shape = (max_boxes_per_image, 5(xmin, ymin, xmax, ymax, class_id))
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
        return image_array, labels_array, image_height, image_width



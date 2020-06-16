import torch
import numpy as np
import cv2
import math
import random

from torch.utils.data import Dataset
from configuration import Config
from utils.iou import IoU


class YoloDataset(Dataset):
    def __init__(self, annotation_dir, transform=None):
        self.annotation_dir = annotation_dir
        self.max_boxes_per_image = Config.max_boxes_per_image
        self.transform = transform

        with open(file=self.annotation_dir, mode="r") as f:
            self.annotation_list = f.readlines()

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


class GroundTruth:
    def __init__(self, device):
        self.device = device
        self.output_feature_sizes = [[Config.input_size[0] // i, Config.input_size[1] // i] for i in Config.yolo_strides]
        self.num_classes = Config.num_classes
        self.anchor_num_per_level = Config.anchor_num_per_level
        self.num_yolo_outputs = Config.num_yolo_outputs
        self.strides = torch.tensor(Config.yolo_strides, dtype=torch.float32, device=device)
        self.anchors = Config.get_anchors().to(self.device)

    def __call__(self, labels, *args, **kwargs):
        """

        :param labels: Tensor, shape: (batch_size, Config.max_boxes_per_image, 5)
        :param args:
        :param kwargs:
        :return:
        """
        batch_size = labels.size()[0]
        batch_label_small = torch.zeros(batch_size, self.output_feature_sizes[0][0],
                                        self.output_feature_sizes[0][1], self.anchor_num_per_level,
                                        5 + self.num_classes, dtype=torch.float32, device=self.device)
        batch_label_middle = torch.zeros(batch_size, self.output_feature_sizes[1][0],
                                         self.output_feature_sizes[1][1], self.anchor_num_per_level,
                                         5 + self.num_classes, dtype=torch.float32, device=self.device)
        batch_label_large = torch.zeros(batch_size, self.output_feature_sizes[2][0],
                                        self.output_feature_sizes[2][1], self.anchor_num_per_level,
                                        5 + self.num_classes, dtype=torch.float32, device=self.device)

        for i in range(batch_size):
            true_label = labels[i]
            true_label = true_label[true_label[..., -1] != -1]
            label = self.__get_true_boxes(boxes=true_label)
            label_small, label_middle, label_large = label
            batch_label_small[i, ...] = label_small
            batch_label_middle[i, ...] = label_middle
            batch_label_large[i, ...] = label_large

        batch_target = [batch_label_small, batch_label_middle, batch_label_large]
        return batch_target



    def __get_true_boxes(self, boxes):
        label = [torch.zeros(self.output_feature_sizes[i][0], self.output_feature_sizes[i][1], self.anchor_num_per_level, 5 + self.num_classes,
                             dtype=torch.float32, device=self.device) for i in range(self.num_yolo_outputs)]
        for i in range(boxes.size()[0]):
            box_xyxy = boxes[i, :4]
            box_xywh = torch.cat(tensors=((box_xyxy[:2] + box_xyxy[2:]) * 0.5, box_xyxy[2:] - box_xyxy[:2]), dim=-1)
            box_xywh_scaled = torch.unsqueeze(box_xywh, dim=0) / torch.unsqueeze(self.strides, dim=1)  # shape: (3, 4)

            box_class = boxes[i, 4].to(torch.int32)
            one_hot = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
            one_hot[box_class] = 1.0

            iou_list = []
            for j in range(self.num_yolo_outputs):
                anchors_xywh = torch.zeros(self.anchor_num_per_level, 4, dtype=torch.float32, device=self.device)
                anchors_xywh[:, 0:2] = torch.floor(box_xywh_scaled[j, 0:2]).to(dtype=torch.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[j] / self.strides[j]

                iou = IoU(box_1=box_xywh_scaled[j], box_2=anchors_xywh).calculate_iou()  # shape: (3, )
                iou_list.append(iou)

            iou_tensor = torch.stack(tensors=iou_list, dim=0)   # shape: (3, 3)
            best_anchor_ind = torch.argmax(iou_tensor.reshape(-1), dim=-1)

            level_idx = best_anchor_ind // self.anchor_num_per_level
            anchor_idx = best_anchor_ind % self.anchor_num_per_level

            x_ind, y_ind = torch.floor(box_xywh_scaled[level_idx, 0:2]).to(torch.int32)
            label[level_idx][y_ind, x_ind, anchor_idx, :] = 0
            label[level_idx][y_ind, x_ind, anchor_idx, 0:4] = box_xywh
            label[level_idx][y_ind, x_ind, anchor_idx, 4:5] = 1.0
            label[level_idx][y_ind, x_ind, anchor_idx, 5:] = one_hot

        return label










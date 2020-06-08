import torch
import numpy as np
import cv2
import math
import random

from torch.utils.data import Dataset
from configuration import Config
from utils.iou import IoU


class YoloDataset(Dataset):
    def __init__(self, transform=None):
        self.annotation_dir = Config.txt_file_dir
        self.batch_size = Config.batch_size
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
        self.strides = torch.tensor(Config.yolo_strides, dtype=torch.float32, device=device)
        self.anchors = Config.get_anchors().reshape(shape=(-1, 2)).to(self.device)

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

        batch_true_label = [batch_label_small, batch_label_middle, batch_label_large]

        center_xy = (labels[..., 0:2] + labels[..., 2:4]) // 2
        wh = labels[..., 2:4] - labels[..., 0:2]
        boxes_xywh = torch.cat(tensors=(center_xy, wh), dim=-1)


        valid_mask = (wh[..., 0] > 0) & (wh[..., 1] > 0)

        for b in range(batch_size):
            valid_boxes = boxes_xywh[b, valid_mask[b]]
            num_valid_boxes = valid_boxes.size()[0]
            if num_valid_boxes == 0:
                continue
            anchors_xywh = torch.cat(tensors=(torch.zeros_like(self.anchors), self.anchors), dim=-1)
            valid_boxes[..., 0:2] = torch.zeros(num_valid_boxes, 2, dtype=torch.float32, device=self.device)
            iou_value = IoU(box_1=torch.unsqueeze(anchors_xywh, dim=0), box_2=torch.unsqueeze(valid_boxes, dim=1)).calculate_iou()

            best_anchor = torch.argmax(iou_value, dim=-1)

            for i, n in enumerate(best_anchor):
                for s in range(self.anchor_num_per_level):
                    if n in Config.anchors_index[s]:
                        x = torch.floor(boxes_xywh[b, i, 0] / self.strides[s]).to(torch.int32)
                        y = torch.floor(boxes_xywh[b, i, 1] / self.strides[s]).to(torch.int32)
                        anchor_id = Config.anchors_index[s].index(n)
                        class_id = labels[b, i, 4].to(torch.int32)
                        batch_true_label[s][b, y, x, anchor_id, 0:4] = boxes_xywh[b, i]
                        batch_true_label[s][b, y, x, anchor_id, 4] = 1.0
                        batch_true_label[s][b, y, x, anchor_id, 5 + class_id] = 1.0

        return batch_true_label





# class GroundTruth:
#     def __init__(self, device):
#         self.device = device
#         self.output_feature_sizes = [[Config.input_size[0] // i, Config.input_size[1] // i] for i in Config.yolo_strides]
#         self.num_classes = Config.num_classes
#         self.anchor_num_per_level = Config.anchor_num_per_level
#         self.max_bbox_per_level = Config.max_bbox_per_level
#         self.strides = torch.tensor(Config.yolo_strides, dtype=torch.float32, device=device)
#         self.anchors = Config.get_anchors()
#
#         self.delta = 0.01
#
#     def __call__(self, labels, *args, **kwargs):
#         batch_size = labels.size()[0]
#         batch_label_small = torch.zeros(batch_size, self.output_feature_sizes[0][0],
#                                         self.output_feature_sizes[0][1], self.anchor_num_per_level,
#                                         5 + self.num_classes, dtype=torch.float32, device=self.device)
#         batch_label_middle = torch.zeros(batch_size, self.output_feature_sizes[1][0],
#                                          self.output_feature_sizes[1][1], self.anchor_num_per_level,
#                                          5 + self.num_classes, dtype=torch.float32, device=self.device)
#         batch_label_large = torch.zeros(batch_size, self.output_feature_sizes[2][0],
#                                         self.output_feature_sizes[2][1], self.anchor_num_per_level,
#                                         5 + self.num_classes, dtype=torch.float32, device=self.device)
#         batch_small_box = torch.zeros(batch_size, self.max_bbox_per_level, 4, dtype=torch.float32, device=self.device)
#         batch_middle_box = torch.zeros(batch_size, self.max_bbox_per_level, 4, dtype=torch.float32, device=self.device)
#         batch_large_box = torch.zeros(batch_size, self.max_bbox_per_level, 4, dtype=torch.float32, device=self.device)
#         for i in range(batch_size):
#             label = labels[i]
#             label = label[label[..., -1] != -1]
#             label, bboxes = self.__get_true_boxes(bboxes=label)
#             label_small, label_middle, label_large = label
#             boxes_small, boxes_middle, boxes_large = bboxes
#
#             batch_label_small[i, ...] = label_small
#             batch_label_middle[i, ...] = label_middle
#             batch_label_large[i, ...] = label_large
#             batch_small_box[i, ...] = boxes_small
#             batch_middle_box[i, ...] = boxes_middle
#             batch_large_box[i, ...] = boxes_large
#
#         small_target = [batch_label_small, batch_small_box]
#         middle_target = [batch_label_middle, batch_middle_box]
#         large_target = [batch_label_large, batch_large_box]
#         return [small_target, middle_target, large_target]
#
#     def __get_true_boxes(self, bboxes):
#         label = [torch.zeros(self.output_feature_sizes[i][0], self.output_feature_sizes[i][1], self.anchor_num_per_level, 5 + self.num_classes,
#                              dtype=torch.float32, device=self.device) for i in range(3)]
#         bboxes_xywh = [torch.zeros(self.max_bbox_per_level, 4, dtype=torch.float32, device=self.device) for _ in range(3)]
#         bboxes_num = torch.zeros(3, dtype=torch.float32, device=self.device)
#         for j in range(bboxes.size()[0]):
#             bbox_coord = bboxes[j, :4]
#             bbox_class_idx = bboxes[j, 4].type(dtype=torch.int32)
#             one_hot = torch.zeros(self.num_classes, dtype=torch.float32, device=self.device)
#             one_hot[bbox_class_idx] = 1.0
#             # uniform_distribution = torch.full(size=(self.num_classes, ), fill_value=1.0 / self.num_classes, dtype=torch.float32, device=self.device)
#             # smooth_one_hot = one_hot * (1 - self.delta) + self.delta * uniform_distribution
#
#             bbox_xywh = torch.cat(tensors=((bbox_coord[2:] + bbox_coord[:2]) * 0.5, bbox_coord[2:] - bbox_coord[:2]), dim=-1)
#             bbox_xywh_scaled = torch.unsqueeze(bbox_xywh, dim=0) / torch.unsqueeze(self.strides, dim=-1)
#
#             iou = []
#             positive_exist = False
#             for i in range(3):
#                 anchors_xywh = torch.zeros(self.anchor_num_per_level, 4, dtype=torch.float32, device=self.device)
#                 anchors_xywh[:, 0:2] = torch.floor(bbox_xywh_scaled[i, 0:2]).to(dtype=torch.int32) + 0.5
#                 anchors_xywh[:, 2:4] = self.anchors[i] / self.strides[i]
#
#                 iou_value = IoU(box_1=torch.unsqueeze(bbox_xywh_scaled[i], dim=0), box_2=anchors_xywh).calculate_iou()
#                 iou.append(iou_value)
#                 iou_mask = iou_value > 0.3
#
#                 if iou_mask.any():
#                     x_ind, y_ind = torch.floor(bbox_xywh_scaled[i, 0:2]).to(torch.int32)
#                     label[i][y_ind, x_ind, iou_mask, :] = 0
#                     label[i][y_ind, x_ind, iou_mask, 0:4] = bbox_xywh
#                     label[i][y_ind, x_ind, iou_mask, 4:5] = 1.0
#                     label[i][y_ind, x_ind, iou_mask, 5:] = one_hot
#
#                     bbox_ind = int(bboxes_num[i] % self.max_bbox_per_level)
#                     bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
#                     bboxes_num[i] += 1
#
#                     positive_exist = True
#
#             if not positive_exist:
#                 iou_tensor = torch.stack(iou, dim=0)
#                 best_anchor_ind = torch.argmax(iou_tensor.reshape(-1), dim=-1)
#                 best_detect = best_anchor_ind // self.anchor_num_per_level
#                 best_anchor = int(best_anchor_ind % self.anchor_num_per_level)
#
#                 x_ind, y_ind = torch.floor(bbox_xywh_scaled[best_detect, 0:2]).to(torch.int32)
#                 label[best_detect][y_ind, x_ind, best_anchor, :] = 0
#                 label[best_detect][y_ind, x_ind, best_anchor, 0:4] = bbox_xywh
#                 label[best_detect][y_ind, x_ind, best_anchor, 4:5] = 1.0
#                 label[best_detect][y_ind, x_ind, best_anchor, 5:] = one_hot
#
#                 bbox_ind = int(bboxes_num[best_detect] % self.max_bbox_per_level)
#                 bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
#                 bboxes_num[best_detect] += 1
# 
#         return label, bboxes_xywh









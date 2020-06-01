import torch
import math


class IoU:
    def __init__(self, box_1, box_2, device):
        """
        The last dimension of box_1 and box_2 are 4(center_x, center_y, w, h)
        :param box_1:
        :param box_2:
        """
        self.box_1 = box_1
        self.box_2 = box_2
        self.device = device

    @staticmethod
    def __get_box_area(box):
        return box[..., 2] * box[..., 3]

    @staticmethod
    def __to_xyxy(box_cxcywh):
        """
        transform to format: (xmin, ymin, xmax, ymax)
        :return:
        """
        box_xyxy = torch.zeros_like(box_cxcywh)
        box_xyxy[..., 0:2] = box_cxcywh[..., 0:2] - 0.5 * box_cxcywh[..., 2:4]
        box_xyxy[..., 2:4] = box_cxcywh[..., 0:2] + 0.5 * box_cxcywh[..., 2:4]
        return box_xyxy

    def calculate_iou(self):
        box_1_area = self.__get_box_area(self.box_1)
        box_2_area = self.__get_box_area(self.box_2)
        box_1_xyxy = IoU.__to_xyxy(self.box_1)
        box_2_xyxy = IoU.__to_xyxy(self.box_2)
        intersect_min = torch.max(box_1_xyxy[..., 0:2], box_2_xyxy[..., 0:2])
        intersect_max = torch.min(box_1_xyxy[..., 2:4], box_2_xyxy[..., 2:4])
        intersect_wh = torch.max(intersect_max - intersect_min, torch.tensor(0.0, device=self.device))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)
        return iou


class GIoU:
    def __init__(self, box_1, box_2, device):
        self.box_1 = GIoU.__fn(GIoU.__to_xyxy(box_1))
        self.box_2 = GIoU.__fn(GIoU.__to_xyxy(box_2))
        self.device = device

    @staticmethod
    def __to_xyxy(box):
        return torch.cat(tensors=(box[..., 0:2] - 0.5 * box[..., 2:4], box[..., 0:2] + 0.5 * box[..., 2:4]), dim=-1)

    @staticmethod
    def __get_area(box):
        return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

    @staticmethod
    def __fn(box):
        return torch.cat(tensors=(torch.min(box[..., 0:2], box[..., 2:4]),
                                  torch.max(box[..., 0:2], box[..., 2:4])), dim=-1)

    def calculate_giou(self):
        box_1_area = GIoU.__get_area(self.box_1)
        box_2_area = GIoU.__get_area(self.box_2)

        intersect_min = torch.max(self.box_1[..., 0:2], self.box_2[..., 0:2])
        intersect_max = torch.min(self.box_1[..., 2:4], self.box_2[..., 2:4])
        intersect_wh = torch.max(intersect_max - intersect_min, torch.tensor(0.0, device=self.device))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area

        enclose_left_up = torch.min(self.box_1[..., 0:2], self.box_2[..., 0:2])
        enclose_right_down = torch.max(self.box_1[..., 2:4], self.box_2[..., 2:4])
        enclose = torch.max(enclose_right_down - enclose_left_up, torch.tensor(0.0, device=self.device))
        enclose_area = enclose[..., 0] * enclose[..., 1]

        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
        return giou


class CIoU:
    def __init__(self, box_1, box_2):
        super(CIoU, self).__init__()
        self.box_1_xywh = box_1
        self.box_2_xywh = box_2
        self.box_1 = CIoU.__to_xyxy(box_1)
        self.box_2 = CIoU.__to_xyxy(box_2)

    @staticmethod
    def __to_xyxy(box):
        return torch.cat(tensors=(box[..., 0:2] - 0.5 * box[..., 2:4], box[..., 0:2] + 0.5 * box[..., 2:4]), dim=-1)

    @staticmethod
    def __get_area(box):
        return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

    @staticmethod
    def __fn(box):
        return torch.cat(tensors=(torch.min(box[..., 0:2], box[..., 2:4]),
                                  torch.max(box[..., 0:2], box[..., 2:4])), dim=-1)

    def calculate_ciou(self):
        box_1_area = CIoU.__get_area(self.box_1)
        box_2_area = CIoU.__get_area(self.box_2)

        intersect_min = torch.max(self.box_1[..., 0:2], self.box_2[..., 0:2])
        intersect_max = torch.min(self.box_1[..., 2:4], self.box_2[..., 2:4])
        intersect_wh = torch.max(intersect_max - intersect_min, torch.zeros_like(intersect_max))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        enclose_left_up = torch.min(self.box_1[..., 0:2], self.box_2[..., 0:2])
        enclose_right_down = torch.max(self.box_1[..., 2:4], self.box_2[..., 2:4])
        enclose_wh = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
        enclose_c_square = torch.sum(torch.pow(enclose_wh[..., 0:2], 2), dim=-1)
        d_square = torch.sum(torch.pow(self.box_1_xywh[..., 0:2] - self.box_2_xywh[..., 0:2], 2), dim=-1)

        v = (4.0 / math.pi ** 2) * torch.pow(torch.atan(self.box_2_xywh[..., 2] / torch.clamp(self.box_2_xywh[..., 3], min=1e-6)) - torch.atan(self.box_1_xywh[..., 2] / torch.clamp(self.box_1_xywh[..., 3], min=1e-6)), 2)
        alpha = v / torch.clamp(1.0 - iou + v, 1e-6)

        ciou = iou - 1.0 * d_square / torch.clamp(enclose_c_square, min=1e-6) - alpha * v

        return ciou

import numpy as np


class IOU(object):
    def __init__(self, box_1, box_2):
        """
        The last dimension of box_1 and box_2 are 4(center_x, center_y, w, h)
        :param box_1:
        :param box_2:
        """
        self.box_1 = np.expand_dims(box_1, axis=0)
        self.box_2 = box_2

    @staticmethod
    def __get_box_area(box):
        return box[..., 2] * box[..., 3]

    @staticmethod
    def __to_xyxy(box_cxcywh):
        """
        transform to format: (xmin, ymin, xmax, ymax)
        :return:
        """
        box_xyxy = np.zeros_like(box_cxcywh)
        box_xyxy[..., 0:2] = box_cxcywh[..., 0:2] - 0.5 * box_cxcywh[..., 2:4]
        box_xyxy[..., 2:4] = box_cxcywh[..., 0:2] + 0.5 * box_cxcywh[..., 2:4]
        return box_xyxy

    def calculate_iou(self):
        box_1_area = self.__get_box_area(self.box_1)
        box_2_area = self.__get_box_area(self.box_2)
        box_1_xyxy = IOU.__to_xyxy(self.box_1)
        box_2_xyxy = IOU.__to_xyxy(self.box_2)
        intersect_min = np.maximum(box_1_xyxy[..., 0:2], box_2_xyxy[..., 0:2])
        intersect_max = np.minimum(box_1_xyxy[..., 2:4], box_2_xyxy[..., 2:4])
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou
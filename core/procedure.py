import torch

from configuration import Config
from core.loss import YoloLoss


class PostProcessing:
    @staticmethod
    def training_procedure(yolo_outputs):
        generate_prediction = GeneratePrediction()
        bboxes = []
        for i, feature in enumerate(yolo_outputs):
            bbox = generate_prediction(feature=feature, feature_index=i)
            bboxes.append(bbox)
        loss_object = YoloLoss()
        loss_value = loss_object(y_pred=bboxes, y_true=None, yolo_outputs=yolo_outputs)

    def testing_procedure(self):
        pass


class GeneratePrediction:
    def __init__(self):
        self.num_classes = Config.num_classes
        self.strides = Config.yolo_strides
        self.anchors = Config.get_anchors()
        self.scale = Config.scale

    @staticmethod
    def __meshgrid(size, B):
        x = torch.arange(start=0, end=size[1], dtype=torch.float32)
        y = torch.arange(start=0, end=size[0], dtype=torch.float32)
        x, y = torch.meshgrid([x, y])
        xy_grid = torch.stack(tensors=(x, y), dim=-1)
        xy_grid = torch.unsqueeze(xy_grid, dim=2)
        xy_grid = torch.unsqueeze(xy_grid, dim=0)
        xy_grid = xy_grid.repeat(B, 1, 1, 3, 1)
        return xy_grid

    def __call__(self, feature, feature_index, *args, **kwargs):
        """
        Generate predictions for one of the features output by yolo.
        :param feature:
        :param feature_index:
        :param args:
        :param kwargs:
        :return: Tensor, size: (batch_size, feature_map_size, feature_map_size, 3, num_classes + 5)
        """
        feature = feature.permute(0, 2, 3, 1)
        shape = feature.size()
        feature = torch.reshape(feature, (shape[0], shape[1], shape[2], 3, -1))
        dx_dy, dw_dh, conf, prob = torch.split(feature, [2, 2, 1, self.num_classes], -1)

        xy_grid = GeneratePrediction.__meshgrid(size=shape[1:3], B=shape[0])

        pred_xy = self.strides[feature_index] * (torch.sigmoid(dx_dy) * self.scale[feature_index] - 0.5 * (self.scale[feature_index] - 1) + xy_grid)
        pred_wh = torch.exp(dw_dh) * self.anchors[feature_index]
        pred_xywh = torch.cat(tensors=(pred_xy, pred_wh), dim=-1)
        pred_conf = torch.sigmoid(conf)
        pred_prob = torch.sigmoid(prob)
        pred_bbox = torch.cat(tensors=(pred_xywh, pred_conf, pred_prob), dim=-1)

        return pred_bbox



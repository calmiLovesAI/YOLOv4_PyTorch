import torch
import torch.nn as nn

from utils.iou import GIoU, IoU
from configuration import Config


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.input_size = torch.tensor(data=Config.input_size, dtype=torch.float32)

    def forward(self, y_pred, y_true, yolo_outputs):
        num_levels = len(yolo_outputs)
        for i in range(num_levels):
            self.__single_level_loss(pred=y_pred[i], feature=yolo_outputs[i], label=y_true[i][0], boxes=y_true[i][1])
        pass

    def __single_level_loss(self, pred, feature, label, boxes):
        N, C, H, W = feature.size()
        feature = torch.reshape(feature, (N, H, W, 3, -1))

        raw_conf = feature[..., 4:5]
        raw_prob = feature[..., 5:]
        pred_xywh = pred[..., 0:4]
        pred_conf = pred[..., 4:5]
        label_xywh = label[..., 0:4]
        respond_bbox = label[..., 4:5]
        label_prob = label[..., 5:]

        giou = torch.unsqueeze(GIoU(box_1=pred_xywh, box_2=label_xywh).calculate_giou(), dim=-1)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (self.input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        # print(boxes.size())  # torch.Size([2, 50, 4])
        pred_xywh = torch.unsqueeze(pred_xywh, dim=4)
        for _ in range(3):
            boxes = torch.unsqueeze(boxes, dim=1)
        iou = IoU(box_1=pred_xywh, box_2=boxes).calculate_iou()
        # print(iou.size())  # torch.Size([2, 52, 52, 3, 50])
        max_iou, indices = torch.max(iou, dim=-1, keepdim=True)
        # print(max_iou.size())    # torch.Size([2, 52, 52, 3, 1])
        

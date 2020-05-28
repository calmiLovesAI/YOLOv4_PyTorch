import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        pass

    def forward(self, y_pred, y_true, yolo_outputs):
        num_levels = len(yolo_outputs)
        for i in range(num_levels):
            self.__single_level_loss(pred=y_pred[i], feature=yolo_outputs[i], label=y_true[i][0], boxes=y_true[i][1])
        pass

    def __single_level_loss(self, pred, feature, label, boxes):
        print(label.size())
        print(boxes.size())
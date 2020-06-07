import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.iou import GIoU, IoU, CIoU
from configuration import Config


class YoloLoss(nn.Module):
    def __init__(self, device):
        super(YoloLoss, self).__init__()
        self.device = device
        self.input_size = torch.tensor(data=Config.input_size, dtype=torch.float32, device=device)
        self.iou_loss_threshold = Config.iou_loss_threshold

        # self.alpha = 0.25
        # self.gamma = 2

    def forward(self, y_pred, y_true, yolo_outputs):
        ciou_loss = 0
        conf_loss = 0
        prob_loss = 0
        num_levels = len(yolo_outputs)
        for i in range(num_levels):
            ciou, conf, prob = self.__single_level_loss(pred=y_pred[i], feature=yolo_outputs[i], label=y_true[i][0], boxes=y_true[i][1])
            ciou_loss += ciou
            conf_loss += conf
            prob_loss += prob
        return ciou_loss, conf_loss, prob_loss

    def __sigmoid_cross_entropy_with_logits(self, labels, logits):
        return torch.max(logits, torch.tensor(0.0, device=self.device)) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))

    def __single_level_loss(self, pred, feature, label, boxes):
        N, C, H, W = feature.size()
        feature = torch.reshape(feature, (N, H, W, 3, -1))

        raw_conf = feature[..., 4:5]
        raw_prob = feature[..., 5:]
        pred_xywh = pred[..., 0:4]
        pred_conf = pred[..., 4:5]
        pred_prob = pred[..., 5:]
        label_xywh = label[..., 0:4]
        respond_bbox = label[..., 4:5]
        label_prob = label[..., 5:]

        ciou = torch.unsqueeze(CIoU(box_1=pred_xywh, box_2=label_xywh).calculate_ciou(), dim=-1)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (self.input_size[0] * self.input_size[1])
        ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

        pred_xywh = torch.unsqueeze(pred_xywh, dim=4)
        for _ in range(3):
            boxes = torch.unsqueeze(boxes, dim=1)
        iou = IoU(box_1=pred_xywh, box_2=boxes).calculate_iou()
        max_iou, indices = torch.max(iou, dim=-1, keepdim=True)
        iou_mask = (max_iou < self.iou_loss_threshold).to(torch.float32)
        respond_bgd = (1.0 - respond_bbox) * iou_mask

        # Focal loss
        # ce = F.binary_cross_entropy_with_logits(raw_conf, respond_bbox)
        # p_t = respond_bbox * pred_conf + respond_bgd * (1 - pred_conf)
        # alpha_factor = respond_bbox * self.alpha + respond_bgd * (1 - self.alpha)
        # modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        # conf_loss = alpha_factor * modulating_factor * ce

        conf_focal = torch.pow(respond_bbox - pred_conf, 2)
        conf_loss = conf_focal * (respond_bbox * F.binary_cross_entropy(pred_conf, respond_bbox)+
                                  respond_bgd * F.binary_cross_entropy(pred_conf, respond_bbox))
        # conf_loss = conf_focal * (respond_bbox * self.__sigmoid_cross_entropy_with_logits(labels=respond_bbox,
        #                                                                                   logits=raw_conf)
        #                           +
        #                           respond_bgd * self.__sigmoid_cross_entropy_with_logits(labels=respond_bbox,
        #                                                                                  logits=raw_conf))
        # prob_loss = respond_bbox * self.__sigmoid_cross_entropy_with_logits(labels=label_prob,
        #                                                                     logits=raw_prob)
        prob_loss = respond_bbox * F.binary_cross_entropy(pred_prob, label_prob)

        ciou_loss = torch.sum(ciou_loss, dim=(1, 2, 3, 4))
        ciou_loss = torch.mean(ciou_loss)
        conf_loss = torch.sum(conf_loss, dim=(1, 2, 3, 4))
        conf_loss = torch.mean(conf_loss)
        prob_loss = torch.sum(prob_loss, dim=(1, 2, 3, 4))
        prob_loss = torch.mean(prob_loss)

        return ciou_loss, conf_loss, prob_loss

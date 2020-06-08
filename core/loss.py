import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.iou import GIoU, IoU, CIoU
from configuration import Config
from core.procedure import GeneratePrediction


class YoloLoss(nn.Module):
    def __init__(self, device):
        super(YoloLoss, self).__init__()
        self.device = device
        self.strides = Config.yolo_strides
        self.anchors = Config.get_anchors().to(device)
        self.input_size = Config.input_size
        self.ignore_threshold = Config.ignore_threshold

    def forward(self, y_pred, y_true):
        reg_loss = 0
        conf_loss = 0
        prob_loss = 0
        num_levels = Config.num_yolo_outputs
        for i in range(num_levels):
            reg, conf, prob = self.__single_level_loss(label=y_true[i], feature=y_pred[i], index=i)
            reg_loss += reg
            conf_loss += conf
            prob_loss += prob
        return reg_loss, conf_loss, prob_loss



    def __meshgrid(self, size, B):
        x = torch.arange(start=0, end=size[1], dtype=torch.float32, device=self.device)
        y = torch.arange(start=0, end=size[0], dtype=torch.float32, device=self.device)
        x, y = torch.meshgrid([x, y])
        xy_grid = torch.stack(tensors=(x, y), dim=-1)
        xy_grid = torch.unsqueeze(xy_grid, dim=2)
        xy_grid = torch.unsqueeze(xy_grid, dim=0)
        xy_grid = xy_grid.repeat(B, 1, 1, 3, 1)
        return xy_grid


    def __single_level_loss(self, label, feature, index):
        N, C, H, W = feature.size()
        feature = feature.permute(0, 2, 3, 1)
        feature_reshaped = torch.reshape(feature, (N, H, W, Config.anchor_num_per_level, -1))
        raw_xywh = feature_reshaped[..., 0:4]
        raw_conf = feature_reshaped[..., 4:5]
        raw_prob = feature_reshaped[..., 5:]
        label_xywh = label[..., 0:4]  # shape: (batch_size, feature_size, feature_size, 3, 4)
        label_conf = label[..., 4:5]  # shape: (batch_size, feature_size, feature_size, 3, 1)
        label_prob = label[..., 5:]  # shape: (batch_size, feature_size, feature_size, 3, 5 + num_classes)

        generate_prediction = GeneratePrediction(self.device)
        pred_bbox = generate_prediction(feature=feature, feature_index=index)
        pred_xywh = pred_bbox[..., 0:4]
        xy_grid = self.__meshgrid(size=(H, W), B=N)
        xy_offset = label_xywh[..., :2] / self.strides[index] - xy_grid # shape: (batch_size, feature_size, feature_size, 3, 2)
        wh_offset = label_conf * torch.log(label_xywh[..., 2:] / self.anchors[index] + Config.avoid_loss_nan_value)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (self.input_size[0] * self.input_size[1])

        ignore_mask_list = []
        for b in range(N):
            true_bboxes = label_xywh[b][label_conf[b][..., 0] != 0.0]
            pred_xywh_b = torch.unsqueeze(pred_xywh[b], dim=3)
            for _ in range(3):
                true_bboxes = torch.unsqueeze(true_bboxes, dim=0)
            iou = IoU(box_1=pred_xywh_b, box_2=true_bboxes).calculate_iou()
            max_iou = torch.zeros(iou.size()[:3], dtype=torch.float32, device=self.device)
            if iou.size()[-1] != 0:
                max_iou, _ = torch.max(iou, dim=-1)
            ignore_mask = max_iou < self.ignore_threshold
            ignore_mask_list.append(ignore_mask.to(torch.float32))
        ignore_mask_tensor = torch.stack(ignore_mask_list, dim=0)
        ignore_mask_tensor = torch.unsqueeze(ignore_mask_tensor, dim=-1)  # shape: (batch_size, feature_size, feature_size, 3, 1)

        xy_loss = label_conf * bbox_loss_scale * F.binary_cross_entropy_with_logits(raw_xywh[..., 0:2], xy_offset)
        wh_loss = label_conf * bbox_loss_scale * 0.5 * torch.square(raw_xywh[..., 2:4] - wh_offset)
        conf_loss = label_conf * F.binary_cross_entropy_with_logits(raw_conf, label_conf) + \
                    (1 - label_conf) * ignore_mask_tensor * F.binary_cross_entropy_with_logits(raw_conf, label_conf)
        prob_loss = label_conf * F.binary_cross_entropy_with_logits(raw_prob, label_prob)

        xy_loss = torch.sum(xy_loss, dim=(1, 2, 3, 4))
        wh_loss = torch.sum(wh_loss, dim=(1, 2, 3, 4))
        reg_loss = torch.mean(xy_loss) + torch.mean(wh_loss)
        conf_loss = torch.sum(conf_loss, dim=(1, 2, 3, 4))
        conf_loss = torch.mean(conf_loss)
        prob_loss = torch.sum(prob_loss, dim=(1, 2, 3, 4))
        prob_loss = torch.mean(prob_loss)

        return reg_loss, conf_loss, prob_loss




# class YoloLoss(nn.Module):
#     def __init__(self, device):
#         super(YoloLoss, self).__init__()
#         self.device = device
#         self.input_size = torch.tensor(data=Config.input_size, dtype=torch.float32, device=device)
#         self.iou_loss_threshold = Config.iou_loss_threshold
#
#         # self.alpha = 0.25
#         # self.gamma = 2
#
#     def forward(self, y_pred, y_true, yolo_outputs):
#         ciou_loss = 0
#         conf_loss = 0
#         prob_loss = 0
#         num_levels = len(yolo_outputs)
#         for i in range(num_levels):
#             ciou, conf, prob = self.__single_level_loss(pred=y_pred[i], feature=yolo_outputs[i], label=y_true[i][0], boxes=y_true[i][1])
#             ciou_loss += ciou
#             conf_loss += conf
#             prob_loss += prob
#         return ciou_loss, conf_loss, prob_loss
#
#     def __sigmoid_cross_entropy_with_logits(self, labels, logits):
#         return torch.max(logits, torch.tensor(0.0, device=self.device)) - logits * labels + torch.log(1 + torch.exp(-torch.abs(logits)))
#
#     def __single_level_loss(self, pred, feature, label, boxes):
#         N, C, H, W = feature.size()
#         feature = torch.reshape(feature, (N, H, W, 3, -1))
#
#         raw_conf = feature[..., 4:5]
#         raw_prob = feature[..., 5:]
#         pred_xywh = pred[..., 0:4]
#         pred_conf = pred[..., 4:5]
#         pred_prob = pred[..., 5:]
#         label_xywh = label[..., 0:4]
#         respond_bbox = label[..., 4:5]
#         label_prob = label[..., 5:]
#
#         ciou = torch.unsqueeze(CIoU(box_1=pred_xywh, box_2=label_xywh).calculate_ciou(), dim=-1)
#
#         bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (self.input_size[0] * self.input_size[1])
#         ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)
#
#         pred_xywh = torch.unsqueeze(pred_xywh, dim=4)
#         for _ in range(3):
#             boxes = torch.unsqueeze(boxes, dim=1)
#         iou = IoU(box_1=pred_xywh, box_2=boxes).calculate_iou()
#         max_iou, indices = torch.max(iou, dim=-1, keepdim=True)
#         iou_mask = (max_iou < self.iou_loss_threshold).to(torch.float32)
#         respond_bgd = (1.0 - respond_bbox) * iou_mask
#
#         # Focal loss
#         # ce = F.binary_cross_entropy_with_logits(raw_conf, respond_bbox)
#         # p_t = respond_bbox * pred_conf + respond_bgd * (1 - pred_conf)
#         # alpha_factor = respond_bbox * self.alpha + respond_bgd * (1 - self.alpha)
#         # modulating_factor = torch.pow(1.0 - p_t, self.gamma)
#         # conf_loss = alpha_factor * modulating_factor * ce
#
#         conf_focal = torch.pow(respond_bbox - pred_conf, 2)
#         conf_loss = conf_focal * (respond_bbox * F.binary_cross_entropy(pred_conf, respond_bbox)+
#                                   respond_bgd * F.binary_cross_entropy(pred_conf, respond_bbox))
#         # conf_loss = conf_focal * (respond_bbox * self.__sigmoid_cross_entropy_with_logits(labels=respond_bbox,
#         #                                                                                   logits=raw_conf)
#         #                           +
#         #                           respond_bgd * self.__sigmoid_cross_entropy_with_logits(labels=respond_bbox,
#         #                                                                                  logits=raw_conf))
#         # prob_loss = respond_bbox * self.__sigmoid_cross_entropy_with_logits(labels=label_prob,
#         #                                                                     logits=raw_prob)
#         prob_loss = respond_bbox * F.binary_cross_entropy(pred_prob, label_prob)
#
#         ciou_loss = torch.sum(ciou_loss, dim=(1, 2, 3, 4))
#         ciou_loss = torch.mean(ciou_loss)
#         conf_loss = torch.sum(conf_loss, dim=(1, 2, 3, 4))
#         conf_loss = torch.mean(conf_loss)
#         prob_loss = torch.sum(prob_loss, dim=(1, 2, 3, 4))
#         prob_loss = torch.mean(prob_loss)
#
#         return ciou_loss, conf_loss, prob_loss

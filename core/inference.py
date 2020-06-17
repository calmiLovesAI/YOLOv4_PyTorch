import torch
import cv2

from configuration import Config
from utils.resize import ResizeTool
from torchvision.ops import nms


class Inference:
    def __init__(self, image_dir, device):
        self.device = device
        self.num_classes = Config.num_classes
        self.strides = Config.yolo_strides
        self.anchors = Config.get_anchors().to(device)
        self.scale = Config.scale
        self.input_size = Config.input_size
        self.score_threshold = Config.score_threshold
        self.nms_iou_threshold  = Config.nms_iou_threshold

        self.image, self.image_size = self.__read_image(image_dir)

    def __call__(self, model, *args, **kwargs):
        feature_maps = model(self.image)
        bboxes = []
        for feature in feature_maps:
            bboxes.append(self.__decode(feature))
        processed_bboxes = self.__process_bbox(bboxes)
        bboxes_tensor, scores_tensor, classes_tensor = self.__filter_bboxes(processed_bboxes)
        indices = nms(boxes=bboxes_tensor, scores=scores_tensor, iou_threshold=self.nms_iou_threshold)
        boxes, scores, classes = bboxes_tensor[indices], scores_tensor[indices], classes_tensor[indices]

        return boxes, scores, classes

    def __read_image(self, image_dir):
        image_array = cv2.imread(filename=image_dir)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_size = image_array.shape[:2]
        image_array = image_array / 255.0
        image_array, _, _, _, _, _ = ResizeTool.resize_image(image_array, Config.input_size)
        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32).to(self.device)
        return image_tensor, image_size

    def __decode(self, feature):
        feature = feature.permute(0, 2, 3, 1)
        shape = feature.size()
        feature = torch.reshape(feature, (shape[0], shape[1], shape[2], 3, -1))
        raw_xywh, raw_conf, raw_prob = torch.split(feature, [4, 1, self.num_classes], -1)
        pred_conf = torch.sigmoid(raw_conf)
        pred_prob = torch.sigmoid(raw_prob)

        return torch.cat(tensors=(raw_xywh, pred_conf, pred_prob), dim=-1)

    def __meshgrid(self, size):
        x = torch.arange(start=0, end=size[1], dtype=torch.float32, device=self.device)
        y = torch.arange(start=0, end=size[0], dtype=torch.float32, device=self.device)
        x, y = torch.meshgrid([x, y])
        xy_grid = torch.stack(tensors=(y, x), dim=-1)
        xy_grid = torch.unsqueeze(xy_grid, dim=2)
        xy_grid = torch.unsqueeze(xy_grid, dim=0)
        xy_grid = xy_grid.repeat(1, 1, 1, 3, 1)
        return xy_grid

    def __process_bbox(self, pred_bboxes):
        for i, pred, in enumerate(pred_bboxes):
            raw_dxdy = pred[:, :, :, :, 0:2]
            raw_dwdh = pred[:, :, :, :, 2:4]

            xy_grid = self.__meshgrid(pred.size()[1:3])

            pred_xy = self.strides[i] * (torch.sigmoid(raw_dxdy) + xy_grid)
            pred_wh = torch.exp(raw_dwdh) * self.anchors[i]

            pred[:, :, :, :,  0:4] = torch.cat(tensors=(pred_xy, pred_wh), dim=-1)

        reshaped_preds = [torch.reshape(bbox, (-1, 5 + self.num_classes)) for bbox in pred_bboxes]
        bboxes = torch.cat(tensors=reshaped_preds, dim=0)
        return bboxes

    def __filter_bboxes(self, bboxes):
        pred_xywh = bboxes[:, 0:4]
        pred_conf = bboxes[:, 4]
        pred_prob = bboxes[:, 5:]

        pred_xyxy = torch.cat(tensors=(pred_xywh[:, :2] - 0.5 * pred_xywh[:, 2:],
                                       pred_xywh[:, :2] + 0.5 * pred_xywh[:, 2:]), dim=-1)
        pred_xyxy = ResizeTool.label_mapping_to_original_image(label=pred_xyxy,
                                                               original_image_size=self.image_size,
                                                               network_input_size=self.input_size)

        pred_xyxy = torch.cat(tensors=(torch.max(pred_xyxy[:, :2], torch.tensor(data=(0, 0), dtype=torch.float32, device=self.device)),
                                       torch.min(pred_xyxy[:, 2:], torch.tensor(data=(self.image_size[1] - 1, self.image_size[0] - 1), dtype=torch.float32, device=self.device))),
                              dim=-1)
        invalid_mask = torch.logical_or(torch.gt(pred_xyxy[:, 0], pred_xyxy[:, 2]), torch.gt(pred_xyxy[:, 1], pred_xyxy[:, 3]))
        pred_xyxy[invalid_mask] = 0

        bbox_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
        area_mask = torch.gt(bbox_area, 0)

        classes = torch.argmax(pred_prob, dim=-1)
        scores = pred_conf * pred_prob[torch.arange(len(pred_xyxy)), classes]
        score_mask = torch.gt(scores, self.score_threshold)

        mask = torch.logical_and(area_mask, score_mask)
        bboxes_tensor, scores_tensor, classes_tensor = pred_xyxy[mask], scores[mask], classes[mask]

        return bboxes_tensor, scores_tensor, classes_tensor




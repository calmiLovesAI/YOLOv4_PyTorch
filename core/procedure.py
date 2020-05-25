import torch

from configuration import Config

class PostProcessing:
    @staticmethod
    def training_procedure(yolo_outputs):
        generate_prediction = GeneratePrediction()
        for i, feature in enumerate(yolo_outputs):
            generate_prediction(feature=feature, feature_index=i)


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
        xy_grid = torch.stack(tensors=(x, y), dim=0)
        xy_grid = torch.unsqueeze(xy_grid, dim=0)
        xy_grid = torch.unsqueeze(xy_grid, dim=0)
        xy_grid = xy_grid.repeat(B, 3, 1, 1, 1)
        return xy_grid

    def __call__(self, feature, feature_index, *args, **kwargs):
        shape = feature.size()
        feature = torch.reshape(feature, (shape[0], 3, -1, shape[2], shape[3]))
        # print(feature.size())
        # torch.Size([2, 3, 25, 52, 52])
        # torch.Size([2, 3, 25, 26, 26])
        # torch.Size([2, 3, 25, 13, 13])
        dx_dy, dw_dh, conf, prob = torch.split(feature, [2, 2, 1, self.num_classes], 2)

        # print(dx_dy.shape, conf.shape)  # torch.Size([2, 3, 2, 52, 52]) torch.Size([2, 3, 1, 52, 52])
        xy_grid = GeneratePrediction.__meshgrid(size=(shape[2], shape[3]), B=shape[0])
        # print(xy_grid.size(), xy_grid.dtype)  # torch.Size([2, 3, 2, 52, 52]) torch.float32
        pred_xy = self.strides[feature_index] * (torch.sigmoid(dx_dy) * self.scale[feature_index] - 0.5 * (self.scale[feature_index] - 1) + xy_grid)
        print("pred_xy形状：", pred_xy.size())  # [2, 3, 2, 52, 52]
        pred_wh = torch.exp(dw_dh) * self.anchors[feature_index]

        pred_xywh = torch.cat(tensors=(pred_xy, pred_wh), dim=1)
        pred_conf = torch.sigmoid(conf)
        pred_prob = torch.sigmoid(prob)
        pred_bbox = torch.cat(tensors=(pred_xywh, pred_conf, pred_prob), dim=1)

        return pred_bbox



import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration import Config
from utils.mish import mish


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn_layer = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn_layer(x)
        if self.activate:
            x = mish(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channel_conv1_in, channel_conv1_out, channel_conv2_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=channel_conv1_in, out_channels=channel_conv1_out,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv2 = ConvBlock(in_channels=channel_conv1_out, out_channels=channel_conv2_out,
                               kernel_size=(3, 3), stride=1, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = residual + x
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.conv2 = ConvBlock(in_channels=out_channels, out_channels=in_channels,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv3 = ConvBlock(in_channels=out_channels, out_channels=in_channels,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        residual_block_list = []
        for i in range(num_residual):
            residual_block_list.append(ResidualBlock(channel_conv1_in=in_channels, channel_conv1_out=in_channels, channel_conv2_out=in_channels))
        self.residual_block = nn.Sequential(*residual_block_list)
        self.conv4 = ConvBlock(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        branch = x
        branch = self.conv2(branch)
        x = self.conv3(x)
        x = self.residual_block(x)
        x = self.conv4(x)
        return torch.cat(tensors=(x, branch), dim=1)


class PoolBlock(nn.Module):
    def __init__(self):
        super(PoolBlock, self).__init__()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x_1 = self.maxpool_1(x)
        x_2 = self.maxpool_2(x)
        x_3 = self.maxpool_3(x)
        return torch.cat(tensors=(x_1, x_2, x_3, x), dim=1)


class CSPDarknet53(nn.Module):
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.conv3 = ConvBlock(in_channels=64, out_channels=64,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv4 = ConvBlock(in_channels=64, out_channels=64,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.residual = ResidualBlock(channel_conv1_in=64, channel_conv1_out=32, channel_conv2_out=64)
        self.conv5 = ConvBlock(in_channels=64, out_channels=64,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv6 = ConvBlock(in_channels=128, out_channels=64,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)

        self.block_1 = BasicBlock(in_channels=64, out_channels=128, num_residual=2)
        self.conv7 = ConvBlock(in_channels=128, out_channels=128,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)

        self.block_2 = BasicBlock(in_channels=128, out_channels=256, num_residual=8)
        self.conv8 = ConvBlock(in_channels=256, out_channels=256,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)

        self.block_3 = BasicBlock(in_channels=256, out_channels=512, num_residual=8)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512,
                               kernel_size=(1, 1), stride=1, padding=0, bias=False)

        self.block_4 = BasicBlock(in_channels=512, out_channels=1024, num_residual=8)
        self.conv10 = ConvBlock(in_channels=1024, out_channels=1024,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv11 = ConvBlock(in_channels=1024, out_channels=512,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv12 = ConvBlock(in_channels=512, out_channels=1024,
                                kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv13 = ConvBlock(in_channels=1024, out_channels=512,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)

        self.pool = PoolBlock()
        self.conv14 = ConvBlock(in_channels=2048, out_channels=512,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv15 = ConvBlock(in_channels=512, out_channels=1024,
                                kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv16 = ConvBlock(in_channels=1024, out_channels=512,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        branch = x
        branch = self.conv3(branch)
        x = self.conv4(x)
        x = self.residual(x)
        x = self.conv5(x)
        x = torch.cat(tensors=(x, branch), dim=1)

        x = self.conv6(x)
        x = self.block_1(x)
        x = self.conv7(x)
        x = self.block_2(x)
        x = self.conv8(x)

        branch_1 = x
        x = self.block_3(x)
        x = self.conv9(x)
        branch_2 = x
        x = self.block_4(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)

        return branch_1, branch_2, x


class YOLOv4(nn.Module):
    def __init__(self):
        super(YOLOv4, self).__init__()
        self.yolo_output_channels = 3 * (Config.num_classes + 5)
        self.backbone = CSPDarknet53()
        self.conv_1 = ConvBlock(in_channels=512, out_channels=256,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.conv_2 = ConvBlock(in_channels=512, out_channels=256,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.convs_1 = YOLOv4.__make_convs(in_channels=512, out_channels=256)

        self.conv_3 = ConvBlock(in_channels=256, out_channels=128,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.conv_4 = ConvBlock(in_channels=256, out_channels=128,
                                kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.convs_2 = YOLOv4.__make_convs(in_channels=256, out_channels=128)

        self.conv_5 = ConvBlock(in_channels=128, out_channels=256,
                                kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv_small= ConvBlock(in_channels=256, out_channels=self.yolo_output_channels,
                                   kernel_size=(1, 1), stride=1, padding=0, bias=True, activate=False)
        self.conv_6 = ConvBlock(in_channels=128, out_channels=256,
                                kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.convs_3 = YOLOv4.__make_convs(in_channels=512, out_channels=256)

        self.conv_7 = ConvBlock(in_channels=256, out_channels=512,
                                kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv_middle = ConvBlock(in_channels=512, out_channels=self.yolo_output_channels,
                                     kernel_size=(1, 1), stride=1, padding=0, bias=True, activate=False)
        self.conv_8 = ConvBlock(in_channels=256, out_channels=512,
                                kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.convs_4 = YOLOv4.__make_convs(in_channels=1024, out_channels=512)

        self.conv_9 = ConvBlock(in_channels=512, out_channels=1024,
                                kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv_large = ConvBlock(in_channels=1024, out_channels=self.yolo_output_channels,
                                    kernel_size=(1, 1), stride=1, padding=0, bias=True, activate=False)

    @staticmethod
    def __make_convs(in_channels, out_channels):
        return nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                      stride=1, padding=0, bias=False),
            ConvBlock(in_channels=out_channels, out_channels=in_channels, kernel_size=(3, 3),
                      stride=1, padding=1, bias=False),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                      stride=1, padding=0, bias=False),
            ConvBlock(in_channels=out_channels, out_channels=in_channels, kernel_size=(3, 3),
                      stride=1, padding=1, bias=False),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                      stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        branch_1, branch_2, x = self.backbone(x)
        layer_1 = x
        x = self.upsample_1(self.conv_1(x))
        branch_2 = self.conv_2(branch_2)
        x = torch.cat(tensors=(x, branch_2), dim=1)
        x = self.convs_1(x)
        layer_2 = x
        x = self.upsample_2(self.conv_3(x))
        branch_1 = self.conv_4(branch_1)
        x = torch.cat(tensors=(x, branch_1), dim=1)
        x = self.convs_2(x)

        layer_3 = x
        x = self.conv_5(x)
        small_bbox = self.conv_small(x)
        x = self.conv_6(layer_3)
        x = torch.cat(tensors=(x, layer_2), dim=1)
        x = self.convs_3(x)

        layer_4 = x
        x = self.conv_7(x)
        middle_bbox = self.conv_middle(x)
        x = self.conv_8(layer_4)
        x = torch.cat(tensors=(x, layer_1), dim=1)
        x = self.convs_4(x)

        x = self.conv_9(x)
        large_bbox = self.conv_large(x)
        return [small_bbox, middle_bbox, large_bbox]

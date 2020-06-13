import torch
import numpy as np

from torchvision import transforms
from PIL import Image
from utils.resize import ResizeTool


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __resize_with_pad(self, image, label):
        resized_image, ratio, top, bottom, left, right = ResizeTool.resize_image(image, self.output_size)
        label = ResizeTool.adjust_label(label, ratio, top, left)

        return resized_image, label

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample["label"]
        image = image / 255.0
        image, label = self.__resize_with_pad(image, label)
        return {
            "image": image,
            "label": label
        }


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample["label"]
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.permute(2, 0, 1)
        label_tensor = torch.from_numpy(label)
        return {
            "image": image_tensor.type(torch.float32),
            "label": label_tensor.type(torch.float32)
        }


class ColorTransform:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample["label"]
        image = Image.fromarray(image)
        image = self.color_jitter(image)
        image = np.array(image)
        return {
            "image": image,
            "label": label
        }
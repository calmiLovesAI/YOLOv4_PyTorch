import torch
import cv2


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample, *args, **kwargs):
        image, label = sample["image"], sample["label"]
        h, w = image.shape[:2]
        ratio = (self.output_size[0] / h, self.output_size[1] / w)
        image = cv2.resize(src=image, dsize=(self.output_size[1], self.output_size[0]), interpolation=cv2.INTER_NEAREST)
        image = image / 255.0
        label[:, 0] = label[:, 0] * ratio[1]
        label[:, 1] = label[:, 1] * ratio[0]
        label[:, 2] = label[:, 2] * ratio[1]
        label[:, 3] = label[:, 3] * ratio[0]
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
            "image": image_tensor,
            "label": label_tensor
        }
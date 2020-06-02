import torch
import cv2


class Rescale:
    def __init__(self, output_size):
        self.output_size = output_size

    def __resize_with_pad(self, image, label):
        h, w = image.shape[:2]
        dst_h, dst_w = self.output_size
        ratio = min(dst_h / h, dst_w / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        top = (dst_h - new_h) // 2
        bottom = dst_h - new_h - top
        left = (dst_w - new_w) // 2
        right = dst_w - new_w - left
        resized_image = cv2.copyMakeBorder(src=image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=128)

        label[:, 0] = label[:, 0] * ratio + left
        label[:, 1] = label[:, 1] * ratio + top
        label[:, 2] = label[:, 2] * ratio + left
        label[:, 3] = label[:, 3] * ratio + top

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
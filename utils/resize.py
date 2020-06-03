import cv2


class ResizeTool:
    @classmethod
    def resize_image(cls, image, output_size):
        h, w = image.shape[:2]
        dst_h, dst_w = output_size
        ratio = min(dst_h / h, dst_w / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)
        top = (dst_h - new_h) // 2
        bottom = dst_h - new_h - top
        left = (dst_w - new_w) // 2
        right = dst_w - new_w - left
        resized_image = cv2.copyMakeBorder(src=image, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=128)
        return resized_image, ratio, top, bottom, left, right

    @classmethod
    def adjust_label(cls, label, ratio, top, left):
        label[:, 0] = label[:, 0] * ratio + left
        label[:, 1] = label[:, 1] * ratio + top
        label[:, 2] = label[:, 2] * ratio + left
        label[:, 3] = label[:, 3] * ratio + top
        return label

    @classmethod
    def label_mapping_to_original_image(cls, label, original_image_size, network_input_size):
        o_h, o_w = original_image_size
        dst_h, dst_w = network_input_size
        ratio = min(dst_h / o_h, dst_w / o_w)
        top_pad = (dst_h - int(o_h * ratio)) // 2
        left_pad = (dst_w - int(o_w * ratio)) // 2
        label[:, 0] = (label[:, 0] - left_pad) / ratio
        label[:, 1] = (label[:, 1] - top_pad) / ratio
        label[:, 2] = (label[:, 2] - left_pad) / ratio
        label[:, 3] = (label[:, 3] - top_pad) / ratio
        return label
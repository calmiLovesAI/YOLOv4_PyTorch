import cv2

from configuration import Config


def draw_boxes_on_image(image, boxes, scores, classes):
    idx2class_dict = Config.get_class_names()
    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        class_and_score = "{}: {:.3f}".format(str(idx2class_dict[classes[i]]), scores[i])
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 2], boxes[i, 3]), color=(250, 206, 135), thickness=2)

        text_size = cv2.getTextSize(text=class_and_score, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        text_width, text_height = text_size[0][0], text_size[0][1]
        cv2.rectangle(img=image, pt1=(boxes[i, 0], boxes[i, 1]), pt2=(boxes[i, 0] + text_width, boxes[i, 1] - text_height), color=(203, 192, 255), thickness=-1)
        cv2.putText(img=image, text=class_and_score, org=(boxes[i, 0], boxes[i, 1] - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    return image


import torch
import cv2
import numpy as np

from core.inference import Inference
from core.yolo_v4 import YOLOv4
from configuration import Config
from utils.visualization import draw_boxes_on_image


def detect_one_picture(model, picture_dir):
    inference = Inference(picture_dir, device)
    boxes, scores, classes = inference(model)
    boxes = boxes.detach().numpy().astype(np.int32)
    scores = scores.detach().numpy().astype(np.float32)
    classes = classes.detach().numpy().astype(np.int32)
    image = draw_boxes_on_image(cv2.imread(filename=picture_dir), boxes, scores, classes)
    return image


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    yolo_v4 = YOLOv4()
    yolo_v4.load_state_dict(torch.load(Config.save_model_dir + "saved_model.pth", map_location=torch.device('cpu')))
    yolo_v4.to(device)
    yolo_v4.eval()

    image = detect_one_picture(yolo_v4, Config.test_single_image_dir)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)



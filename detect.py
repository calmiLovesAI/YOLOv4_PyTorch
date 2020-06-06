import torch
import cv2
import numpy as np

from core.inference import Inference
from core.yolo_v4 import YOLOv4
from configuration import Config
from utils.visualization import draw_boxes_on_image


def detect_one_picture(model, picture_dir, device):
    inference = Inference(picture_dir, device)
    with torch.no_grad():
        boxes, scores, classes = inference(model)
    boxes = boxes.cpu().numpy().astype(np.int32)
    scores = scores.cpu().numpy().astype(np.float32)
    classes = classes.cpu().numpy().astype(np.int32)
    image = draw_boxes_on_image(cv2.imread(filename=picture_dir), boxes, scores, classes)
    return image


def detect_multiple_pictures(model, pictures, epoch, device):
    index = 0
    for picture in pictures:
        index += 1
        result = detect_one_picture(model=model, picture_dir=picture, device=device)
        cv2.imwrite(filename=Config.training_results_save_dir + "epoch-{}-picture-{}.jpg".format(epoch, index), img=result)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    yolo_v4 = YOLOv4()
    if Config.detect_on_cpu:
        yolo_v4.load_state_dict(torch.load(Config.save_model_dir + "saved_model.pth", map_location=torch.device('cpu')))
    else:
        yolo_v4.load_state_dict(torch.load(Config.save_model_dir + "saved_model.pth"))
    yolo_v4.to(device)
    yolo_v4.eval()

    image = detect_one_picture(yolo_v4, Config.test_single_image_dir, device)

    cv2.namedWindow("detect result", flags=cv2.WINDOW_NORMAL)
    cv2.imshow("detect result", image)
    cv2.waitKey(0)



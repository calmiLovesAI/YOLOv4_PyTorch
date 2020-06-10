import torch


class Config:
    epochs = 50
    batch_size = 6

    input_size = (416, 416)

    # save model
    save_model_dir = "./saved_model/"
    save_frequency = 20
    load_weights_before_training = False
    load_weights_from_epoch = 0

    # test image
    test_single_image_dir = ""
    test_images_during_training = False
    training_results_save_dir = "./test_pictures/"
    test_images_dir_list = ["", ""]

    detect_on_cpu = True

    # network structure
    yolo_strides = [8, 16, 32]
    yolo_anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    anchors_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    scale = [1.2, 1.1, 1.05]
    anchor_num_per_level = 3
    num_yolo_outputs = len(yolo_strides)

    # dataset
    num_classes = 20
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"

    pascal_voc_classes = {0: "person", 1: "bird", 2: "cat", 3: "cow", 4: "dog", 5: "horse",
                          6: "sheep", 7: "aeroplane", 8: "bicycle", 9: "boat", 10: "bus",
                          11: "car", 12: "motorbike", 13: "train", 14: "bottle", 15: "chair",
                          16: "diningtable", 17: "pottedplant", 18: "sofa", 19: "tvmonitor"}

    txt_file_dir = "data.txt"
    max_boxes_per_image = 50
    max_bbox_per_level = 150

    ignore_threshold = 0.5

    avoid_loss_nan_value = 1e-10

    score_threshold = 0.5
    nms_iou_threshold = 0.2


    @classmethod
    def get_anchors(cls):
        return torch.tensor(cls.yolo_anchors, dtype=torch.float32).reshape(3, 3, 2)

    @classmethod
    def get_class_names(cls, from_file=False, file_dir=None):
        if from_file:
            with open(file=file_dir, mode="r", encoding="utf-8") as f:
                class_names = dict((i, name.strip("\n")) for i, name in enumerate(f.readlines()))
        else:
            class_names = cls.pascal_voc_classes
        return class_names

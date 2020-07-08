import torch


class Config:
    epochs = 50
    batch_size = 6

    input_size = (416, 416)

    initial_learning_rate = 1e-3
    MultiStepLR_milestones = [500, 2000]

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

    # train set and valid set
    txt_file_dir = "data.txt"
    valid_ratio = 0.1
    train_txt = "./train.txt"
    valid_txt = "./valid.txt"

    # network structure
    yolo_strides = [8, 16, 32]
    yolo_anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    anchors_file = "anchors.txt"
    anchors_from_file = True
    scale = [1.2, 1.1, 1.05]
    anchor_num_per_level = 3
    num_yolo_outputs = len(yolo_strides)

    # dataset
    dataset_type = "coco"  # "voc" or "coco"
    num_classes = 80   # 20 for voc and 80 for coco
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"

    pascal_voc_classes = {0: "person", 1: "bird", 2: "cat", 3: "cow", 4: "dog", 5: "horse",
                          6: "sheep", 7: "aeroplane", 8: "bicycle", 9: "boat", 10: "bus",
                          11: "car", 12: "motorbike", 13: "train", 14: "bottle", 15: "chair",
                          16: "diningtable", 17: "pottedplant", 18: "sofa", 19: "tvmonitor"}

    coco_root = "./data/datasets/COCO/2017/"
    coco_images = coco_root + "train2017/"
    coco_annotations = coco_root + "annotations/instances_train2017.json"

    coco_classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
                    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
                    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
                    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
                    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
                    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
                    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
                    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
                    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
                    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
                    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
                    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
                    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
                    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
                    78: "hair drier", 79: "toothbrush"}

    class_file_dir = ""
    class_from_file = False

    max_boxes_per_image = 150

    ignore_threshold = 0.5

    avoid_loss_nan_value = 1e-6

    score_threshold = 0.5
    nms_iou_threshold = 0.2


    @classmethod
    def get_anchors(cls):
        if cls.anchors_from_file:
            with open(file=cls.anchors_file, mode="r", encoding="utf-8") as f:
                anchors_str = f.readline()
            anchors_list = anchors_str.split(", ")
            anchors = [float(i) for i in anchors_list]
            return torch.tensor(anchors, dtype=torch.float32).reshape(3, 3, 2)
        else:
            return torch.tensor(cls.yolo_anchors, dtype=torch.float32).reshape(3, 3, 2)

    @classmethod
    def class2idx(cls):
        return dict((v, k) for k, v in Config.get_class_names().items())

    @classmethod
    def get_class_names(cls):
        if cls.class_from_file:
            with open(file=cls.class_file_dir, mode="r", encoding="utf-8") as f:
                class_names = dict((i, name.strip("\n")) for i, name in enumerate(f.readlines()))
        else:
            if cls.dataset_type == "voc":
                class_names = cls.pascal_voc_classes
            elif cls.dataset_type == "coco":
                class_names = cls.coco_classes
            else:
                raise ValueError("Wrong dataset_type!")
        return class_names

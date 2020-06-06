import torch


class Config:
    epochs = 50
    batch_size = 5

    input_size = (416, 416)

    # save model
    save_model_dir = "./saved_model/"
    save_frequency = 10
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
    scale = [1.2, 1.1, 1.05]
    anchor_num_per_level = 3

    # dataset
    num_classes = 20
    pascal_voc_root = "./data/datasets/VOCdevkit/VOC2012/"
    pascal_voc_images = pascal_voc_root + "JPEGImages"
    pascal_voc_labels = pascal_voc_root + "Annotations"

    pascal_voc_classes = {"person": 0, "bird": 1, "cat": 2, "cow": 3, "dog": 4,
                          "horse": 5, "sheep": 6, "aeroplane": 7, "bicycle": 8,
                          "boat": 9, "bus": 10, "car": 11, "motorbike": 12,
                          "train": 13, "bottle": 14, "chair": 15, "diningtable": 16,
                          "pottedplant": 17, "sofa": 18, "tvmonitor": 19}

    txt_file_dir = "data.txt"
    max_boxes_per_image = 50
    max_bbox_per_level = 150

    iou_loss_threshold = 0.5

    avoid_loss_nan_value = 1e-10

    score_threshold = 0.6
    nms_iou_threshold = 0.2


    @classmethod
    def get_anchors(cls):
        return torch.tensor(cls.yolo_anchors, dtype=torch.float32).reshape(3, 3, 2)

    @classmethod
    def idx2class(cls):
        return dict((v, k) for k, v in Config.pascal_voc_classes.items())

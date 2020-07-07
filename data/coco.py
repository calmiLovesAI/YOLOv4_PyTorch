from configuration import Config
import json
from pathlib import Path
import time
import cv2


class COCO:
    def __init__(self):
        self.annotation_dir = Config.coco_annotations
        self.images_dir = Config.coco_images
        self.train_annotation = Path(self.annotation_dir)
        start_time = time.time()
        self.train_dict = COCO.__load_json(self.train_annotation)
        print("It took {:.2f} seconds to load the json files.".format(time.time() - start_time))
        print(self.__get_category_id_information(self.train_dict))

    @staticmethod
    def __load_json(json_file):
        print("Start loading {}...".format(json_file.name))
        with json_file.open(mode='r') as f:
            load_dict = json.load(f)
        print("Loading is complete!")
        return load_dict

    @staticmethod
    def __find_all(x, value):
        list_data = []
        for i in range(len(x)):
            if x[i] == value:
                list_data.append(i)
        return list_data

    @staticmethod
    def __get_image_information(data_dict):
        images = data_dict["images"]
        image_file_list = []
        image_id_list = []
        image_height_list = []
        image_width_list = []
        for image in images:
            image_file_list.append(image["file_name"])
            image_id_list.append(image["id"])
            image_height_list.append(image["height"])
            image_width_list.append(image["width"])
        return image_file_list, image_id_list, image_height_list, image_width_list

    @staticmethod
    def __get_bounding_box_information(data_dict):
        annotations = data_dict["annotations"]
        image_id_list = []
        bbox_list = []
        category_id_list = []
        for annotation in annotations:
            category_id_list.append(annotation["category_id"])
            image_id_list.append(annotation["image_id"])
            bbox_list.append(annotation["bbox"])
        return image_id_list, bbox_list, category_id_list

    @staticmethod
    def __get_category_id_information(data_dict):
        categories = data_dict["categories"]
        category_dict = {}
        for category in categories:
            category_dict[category["name"]] = category["id"]
        return category_dict

    def __bbox_information(self, image_id, image_ids_from_annotation, bboxes, image_height, image_width, category_ids):
        processed_bboxes = []
        index_list = COCO.__find_all(x=image_ids_from_annotation, value=image_id)
        for index in index_list:
            x, y, w, h = bboxes[index]
            x_min, y_min = int(x), int(y)
            x_max = int(x + w)
            y_max = int(y + h)
            processed_bboxes.append([x_min, y_min, x_max, y_max, self.__category_id_transform(category_ids[index])])
        return processed_bboxes

    def __category_id_transform(self, original_id):
        category_id_dict = COCO.__get_category_id_information(self.train_dict)
        id2category = dict((v, k) for k, v in category_id_dict.items())
        return Config.class2idx()[id2category[original_id]]

    @staticmethod
    def __bbox_str(bboxes):
        bbox_info = ""
        for bbox in bboxes:
            for item in bbox:
                bbox_info += str(item)
                bbox_info += " "
        return bbox_info.strip()

    @staticmethod
    def __get_image_h_and_w(image_dir):
        image_array = cv2.imread(image_dir)
        h, w = image_array.shape[0:2]
        return int(h), int(w)

    def write_data_to_txt(self, txt_dir):
        image_files, image_ids, image_heights, image_widths = COCO.__get_image_information(self.train_dict)
        image_ids_from_annotation, bboxes, category_ids = COCO.__get_bounding_box_information(self.train_dict)
        with open(file=txt_dir, mode="a+") as f:
            picture_index = 0
            for i in range(len(image_files)):
                write_line_start_time = time.time()
                image_dir = self.images_dir + image_files[i]
                h, w = COCO.__get_image_h_and_w(image_dir)
                line_info = ""
                line_info += image_dir + " " + str(h) + " " + str(w) + " "
                processed_bboxes = self.__bbox_information(image_ids[i],
                                                           image_ids_from_annotation,
                                                           bboxes,
                                                           image_heights[i],
                                                           image_widths[i],
                                                           category_ids)
                if processed_bboxes:
                    picture_index += 1
                    line_info += COCO.__bbox_str(bboxes=processed_bboxes)
                    line_info += "\n"
                    print("Writing information of the {}th picture {} to {}, which took {:.2f}s".format(picture_index, image_files[i], txt_dir, time.time() - write_line_start_time))
                    f.write(line_info)




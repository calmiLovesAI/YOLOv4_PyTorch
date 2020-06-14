from data.voc import VOC
from configuration import Config


def generate_txt(dataset):
    with open(file=Config.txt_file_dir, mode="a+", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            num_bboxes = len(sample["bboxes"])
            line_text = sample["image_file_dir"] + " " + str(sample["image_height"]) + " " + str(sample["image_width"]) + " "
            for j in range(num_bboxes):
                bbox = list(map(str, sample["bboxes"][j]))
                cls = str(sample["class_ids"][j])
                bbox.append(cls)
                line_text += " ".join(bbox)
                line_text += " "
            line_text = line_text.strip()
            line_text += "\n"
            print("Writing information of picture {} to {}".format(sample["image_file_dir"], Config.txt_file_dir))
            f.write(line_text)


if __name__ == '__main__':
    if Config.dataset_type == "voc":
        dataset = VOC()
        generate_txt(dataset=dataset)

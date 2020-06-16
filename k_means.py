import numpy as np

from configuration import Config


class KMeans:
    def __init__(self, label_dir, anchor_num):
        self.label_file = label_dir
        self.anchor_num = anchor_num

    def generate_anchors(self):
        boxes = self.load_data()
        anchors = KMeans.k_means(boxes, self.anchor_num)
        anchors = anchors[np.argsort(anchors[:, 0])]
        acc = KMeans.average_iou(boxes, anchors)
        print("Accuracy: {:.2f}%".format(acc * 100))
        anchors = anchors * np.array(Config.input_size[::-1])
        return anchors

    @staticmethod
    def average_iou(boxes, cluster):
        return np.mean([np.max(KMeans.get_iou(boxes[i], cluster)) for i in range(boxes.shape[0])])

    @staticmethod
    def k_means(boxes, k):
        row = boxes.shape[0]
        distance = np.empty((row, k))
        last_cluster = np.zeros((row, ))
        # np.random.seed()
        cluster = boxes[np.random.choice(row, k, replace=False)]
        while True:
            for i in range(row):
                distance[i] = 1 - KMeans.get_iou(boxes[i], cluster)

            near = np.argmin(distance, axis=1)

            if(last_cluster == near).all():
                break

            for j in range(k):
                cluster[j] = np.median(boxes[near == j], axis=0)

            last_cluster = near
        return cluster


    def load_data(self):
        boxes = []
        with open(file=self.label_file, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            pieces = line.strip().split(" ")
            h, w = float(pieces[1]), float(pieces[2])
            num_objects = (len(pieces) - 3) // 5
            for i in range(num_objects):
                xmin = float(pieces[3 + i * 5]) / w
                ymin = float(pieces[3 + i * 5 + 1]) / h
                xmax = float(pieces[3 + i * 5 + 2]) / w
                ymax = float(pieces[3 + i * 5 + 3]) / h

                xmin = np.float64(xmin)
                ymin = np.float64(ymin)
                xmax = np.float64(xmax)
                ymax = np.float64(ymax)

                boxes.append([xmax - xmin, ymax - ymin])

        return np.array(boxes)

    @staticmethod
    def get_iou(box, cluster):
        x = np.minimum(box[0], cluster[:, 0])
        y = np.minimum(box[1], cluster[:, 1])

        inter = x * y
        area_box = box[0] * box[1]
        area_cluster = cluster[:, 0] * cluster[:, 1]
        iou = inter / (area_box + area_cluster - inter)
        return iou



if __name__ == '__main__':
    anchors = KMeans(label_dir=Config.txt_file_dir, anchor_num=9).generate_anchors()
    print(anchors)
    anchors_str = ""
    for i in range(anchors.shape[0]):
        if i:
            anchors_str += ", "
        anchors_str += str(int(anchors[i][0])) + ", " + str(int(anchors[i][1]))
    with open(file=Config.anchors_file, mode="a+", encoding="utf-8") as f:
        f.write(anchors_str)
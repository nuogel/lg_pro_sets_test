import numpy as np
import glob
import xml.etree.ElementTree as ET


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


if __name__ == '__main__':

    def load_dataset_xml(path):
        dataset = []
        for xml_file in glob.glob("{}/*xml".format(path)):
            tree = ET.parse(xml_file)

            height = 1  # int(tree.findtext("./size/height"))
            width = 1  # int(tree.findtext("./size/width"))

            for obj in tree.iter("object"):
                xmin = float(obj.findtext("bndbox/xmin")) / width
                ymin = float(obj.findtext("bndbox/ymin")) / height
                xmax = float(obj.findtext("bndbox/xmax")) / width
                ymax = float(obj.findtext("bndbox/ymax")) / height
                if xmax - xmin>0 and ymax - ymin>0:
                    dataset.append([xmax - xmin, ymax - ymin])

        return np.array(dataset)


    def load_dataset_txt(path):
        dataset = []
        for txt_file in glob.glob("{}/*.txt".format(path)):
            height = 1  # int(tree.findtext("./size/height"))
            width = 1  # int(tree.findtext("./size/width"))
            lines = open(txt_file, 'r').readlines()
            for line in lines:
                tmps = line.split(',')
                w = float(tmps[2])
                h = float(tmps[3])
                if w==0.or h==0.:
                    print('bad bbox:', txt_file)
                    continue
                dataset.append([w, h])
        return np.array(dataset)


    ANNOTATIONS_PATH ='/media/dell/data/比赛/第一批标注数据35张/labels'
    # 'E:/datasets/VisDrone2019/VisDrone2019-DET-train/annotations/'
    CLUSTERS = 9

    data = load_dataset_xml(ANNOTATIONS_PATH)
    # data = load_dataset_txt(ANNOTATIONS_PATH)
    out = kmeans(data, k=CLUSTERS)
    out = np.asarray(out)

    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    Area = (out[:, 0] * out[:, 1])
    idx = np.argsort(Area)
    Boxout = out[idx]

    Area = (Boxout[:, 0] * Boxout[:, 1])
    print("Area:\n {}".format(Area))

    ratios = np.around(Boxout[:, 0] / Boxout[:, 1], decimals=2)
    print("Ratios:\n {}".format(ratios))
    print(Boxout[::-1])


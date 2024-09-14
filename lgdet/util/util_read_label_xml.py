import xml.etree.ElementTree as ET


def _read_label_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        cls = obj.find('name').text

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        _bbox = (float(bbox.find('ymin').text) / shape[0], float(bbox.find('xmin').text) / shape[1],
                 float(bbox.find('ymax').text) / shape[0], float(bbox.find('xmax').text) / shape[1])
        bboxes.append([cls, _bbox])
    return bboxes, shape, difficult, truncated

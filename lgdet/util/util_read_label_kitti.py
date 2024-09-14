def _read_line(path, pass_obj=['Others', ]):
    """
    Parse the labels from file.

    :param pass_obj: pass the labels in the list.
                    e.g. pass_obj=['Others','Pedestrian']
    :param path: the path of file that need to parse.
    :return:lists of the classes and the key points.
    """
    file_open = open(path, 'r')
    bbs = []
    for line in file_open.readlines():
        tmps = line.strip().split(' ')
        if tmps[0] in pass_obj:
            continue
        box_x1 = float(tmps[4])
        box_y1 = float(tmps[5])
        box_x2 = float(tmps[6])
        box_y2 = float(tmps[7])
        bbs.append([tmps[0], box_x1, box_y1, box_x2, box_y2])
    return bbs
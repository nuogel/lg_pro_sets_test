import cv2

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",
             "O"]

ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def _read_CCPD_title(filename):
    tmp = filename.split('.')[0].split('-')
    assert len(tmp) == 7, 'filename is not correct.'
    _area, _degree, _bbox, _corner, _license_number, _luminance, _blur = tmp
    area = float(_area)
    degree = [float(_degree.split('_')[i]) for i in range(2)]
    bbox = [int(_bbox.split('_')[i].split('&')[j]) for i in range(2) for j in range(2)]
    corner = [int(_corner.split('_')[i].split('&')[j]) for i in range(4) for j in range(2)]
    _license_number = _license_number.split('_')
    licensenumber = [provinces[int(_license_number[0])]] + [ads[int(_license_number[i])] for i in range(1, 7)]
    luminance = float(_luminance)
    blur = float(_blur)
    return area, degree, bbox, corner, licensenumber, luminance, blur


def _show_ccpd(file, canny=False, crop_licience_plante=False):
    from others.CV_Traditional.carPlante_bianyuan import car_canny
    lines = open(file, 'r').readlines()
    for line in lines:
        filename = line.split(';')[0].strip()
        img_path = line.split(';')[2].strip()
        img = cv2.imread(img_path)
        img_raw = img.copy()
        info = _read_CCPD_title(filename)
        x1 = info[2][0]
        y1 = info[2][1]
        x2 = info[2][2]
        y2 = info[2][3]
        # img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0))
        if crop_licience_plante:
            img = _crop_licience_plante(img, filename)
            img = cv2.resize(img, (880, 400), interpolation=cv2.INTER_CUBIC)

        if canny:
            car_canny(img)
        cv2.imshow('img', img_raw)
        cv2.waitKey()
        cv2.destroyAllWindows()




def _crop_licience_plante(img, filename):
    info = _read_CCPD_title(filename)
    x1, y1, x2, y2 = info[2]
    img = img[y1:y2, x1:x2, :]
    return img



if __name__ == '__main__':
    file = 'E:/LG/GitHub/lg_pro_sets/util/util_tmp/make_list.txt'
    filename = '025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg'
    # bbx = _read_CCPD_title(filename)
    _show_ccpd(file, canny=True, crop_licience_plante=True)

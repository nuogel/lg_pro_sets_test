import cv2


def img_cut(h, w, gap=200, target_size=(512, 768)):
    '''
    output:[x1, y1, x2, y2]
    :param h:
    :param w:
    :param gap:
    :param target_size:
    :return:
    '''
    t_w = target_size[1]
    t_h = target_size[0]

    H_OK = False
    img_cuts_pixcel = []
    cut_ha = 0
    cut_hb = t_h
    while 1:
        cut_wa = 0
        cut_wb = t_w
        W_OK = False
        while 1:
            img_cuts_pixcel.append([cut_wa, cut_ha, cut_wb, cut_hb])
            if W_OK:
                break
            cut_wa = cut_wb - gap
            cut_wb = cut_wa + t_w

            if cut_wb > w:
                cut_wb = w
                cut_wa = cut_wb - t_w
                W_OK = True
        if H_OK:
            break
        cut_ha = cut_hb - gap
        cut_hb = cut_ha + t_h

        if cut_hb > h:
            cut_hb = h
            cut_ha = cut_hb - t_h
            H_OK = True

    return img_cuts_pixcel


if __name__ == '__main__':
    img = cv2.imread('F:/Projects/auto_Airplane/TS02/20191217_153659/894.png')
    cv2.imshow('img_', cv2.resize(img, None, fx=0.8, fy=0.8))
    cv2.waitKey()

    [h, w, c] = img.shape
    gap = 200
    target_size = (512, 768)
    img_cuts_pixcel = img_cut(h, w, gap=200, target_size=(512, 768))
    for bbox in img_cuts_pixcel:
        img_cut = img[bbox[1]: bbox[3], bbox[0]:bbox[2]]
        cv2.imshow('img', img_cut)
        cv2.waitKey()

import cv2


def resize_save(img_path, save_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_resize = cv2.resize(img, (w // 4, h // 4))
    cv2.imshow('img', img_resize)
    cv2.waitKey()
    cv2.imwrite(save_path, img_resize)


if __name__ == '__main__':
    img_path = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/saved/yolov3_kitti_384x960.png'
    save_path = '00000_1_4X.png'
    resize_save(img_path, save_path)

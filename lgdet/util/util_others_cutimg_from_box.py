import cv2
import os

imgpath = '/media/dell/data/ocr/电表识别/123_combine/images'
labpath = '/media/dell/data/ocr/电表识别/123_combine/labels'
save_path= os.path.join(imgpath, '..', 'cut_images')

for imgp in os.listdir(imgpath):
    imgp_i = os.path.join(imgpath, imgp)
    img = cv2.imread(imgp_i)
    labp_i = os.path.join(labpath, imgp.replace('.jpg', '.txt'))
    f = open(labp_i)
    for i, line in enumerate(f.readlines()):
        tmp = line.strip().split(',')
        name = tmp[-2]
        x = list(map(int,tmp[:-3][::2]))
        y =list(map(int, tmp[:-2][1::2]))
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        try:
            img_cut = img[ymin:ymax, xmin:xmax, :]
            save_name = os.path.join(save_path, name+'_'+imgp.split('.')[0]+'_'+str(i)+'.jpg')
            cv2.imwrite(save_name, img_cut)
        except:
            pass





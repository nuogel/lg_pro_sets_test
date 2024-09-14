import cv2


def Jpegcompress2(src, quality):
    params = []
    # /*IMWRITE_JPEG_QUALITY For JPEG, it can be a quality from 0 to 100
    # (the higher is the better). Default value is 95 */
    params.append(cv2.IMWRITE_JPEG_QUALITY)
    params.append(quality)
    # //将图像压缩编码到缓冲流区域
    buff = cv2.imencode(".jpg", src, params)[1]
    # print(len(buff))
    # //将压缩后的缓冲流内容解码为Mat，进行后续的处理
    dest = cv2.imdecode(buff, cv2.IMREAD_COLOR)

    return dest


if __name__ == '__main__':
    imgpath = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/1.png'
    img = cv2.imread(imgpath)
    img_mosaic = Jpegcompress2(img, 10)
    cv2.imwrite('E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/img_mosaic.png', img_mosaic)
    cv2.imshow('img', img_mosaic)
    cv2.waitKey()

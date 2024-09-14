from others.cv_traditional.feature_matching.sift import *
from others.cv_traditional.feature_matching.surf import *
from others.cv_traditional.feature_matching.orb import *


def compare(filename):
    imgs = []

    keyPoint = []
    descriptor = []
    img, keyPoint_temp, descriptor_temp = sift(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    img, keyPoint_temp, descriptor_temp = surf(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    img, keyPoint_temp, descriptor_temp = orb(filename)
    keyPoint.append(keyPoint_temp)
    descriptor.append(descriptor_temp)
    imgs.append(img)
    return imgs, keyPoint, descriptor


def match(filename1, filename2, method):
    if (method == 'sift'):
        img1, kp1, des1 = sift(filename1)
        img2, kp2, des2 = sift(filename2)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # sift的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    if (method == 'surf'):
        img1, kp1, des1 = surf(filename1)
        img2, kp2, des2 = surf(filename2)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # surf的normType应该使用NORM_L2或者NORM_L1
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    if (method == 'orb'):
        img1, kp1, des1 = orb(filename1)
        img2, kp2, des2 = orb(filename2)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # orb的normType应该使用NORM_HAMMING
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        knnMatches = bf.knnMatch(des1, des2, k=1)  # drawMatchesKnn
    for m in matches:
        for n in matches:
            if (m.distance >= n.distance * 0.75):
                matches.remove(m)
                break
    # print('%s size of kp: %d, after filtering: %d' % (method, len(des1), len(matches)))
    img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img2, flags=2)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.imwrite('out.png', img)
    cv2.destroyAllWindows()


def main():
    methods = ['sift'] #, 'surf', 'orb']
    for method in methods:
        match('F:/Projects/auto_Airplane/1f_2.png', 'F:/Projects/auto_Airplane/f1_l.png', method)


if __name__ == '__main__':
    main()

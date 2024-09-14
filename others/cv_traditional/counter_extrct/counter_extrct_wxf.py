import os
import random

import cv2
import librosa
import numpy as np
from sklearn.cluster import KMeans
imgpath = '/home/dell/lg/code/lg_pro_sets/others/cv_traditional/counter_extrct/image'

for imgpi in os.listdir(imgpath)[3:]:
    img = cv2.imread(os.path.join(imgpath, imgpi))
    h,w,c = img.shape
    img = img[1000:1800,:,:]
    cv2.imshow('img', img)
    # cv2.waitKey()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    cv2.imshow('imgg', img_gray)
    # cv2.waitKey()

    threshcount = img_gray[:, 600]
    wall = range(500, 800, 2)

    pix = img_gray[:, wall].reshape(-1)
    pix = pix[pix<240]
    pix = pix[pix>10]
    pixdict = {}
    pixset = list(set(pix))
    for pi in pixset:
        pixdict[pi] = list(pix).count(pi)

    pixdictsort = sorted(pixdict.items(), key=lambda x:x[1], reverse=True)

    pixall = []
    for pix10 in pixdictsort[:10]:
        flag=False
        k,v = pix10
        if pixall == []:
            pixall.append([k])
        else:
            ll = len(pixall)
            for i in range(ll):
                if abs(np.asarray(pixall[i]).mean()-k)<=5:
                    pixall[i].append(k)
                    flag = True
                    break

            if not flag:
                pixall.append([k])
    threshdict = {}
    for p in pixall:
        count = 0
        for pi in p:
            count+=pixdict[pi]
        threshdict[int(np.asarray(p).mean())] = count
    threshcenter = [10]
    threshdict = sorted(threshdict.items(), key=lambda x:x[1], reverse=True)
    for i in range(2):
        threshcenter.append(threshdict[i][0])




    # km = KMeans(n_clusters=2, random_state=1).fit(np.asarray(pix).reshape(-1, 1))
    #
    # labels = km.labels_
    # center = km.cluster_centers_
    # threshcenter = center.reshape(-1)
    # threshcenter = sorted([int(x) for x in threshcenter])[:3]

    for thresh in threshcenter:
        img_b = cv2.threshold(img_gray, thresh+5, 255,  cv2.THRESH_BINARY)[1] # >thresh ->255
        cv2.imshow('img', img_b)
        # cv2.waitKey()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        img_b = cv2.morphologyEx(img_b, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('img', img_b)
        cv2.waitKey()

        contours, hierarchy = cv2.findContours(img_b,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key= lambda x:x.shape[0])
        img = cv2.drawContours(img,contours[-1],-1,(0,0,255),1)
        cv2.imshow('img', img)
        cv2.waitKey()

    cv2.imwrite('../wxf.png', img)


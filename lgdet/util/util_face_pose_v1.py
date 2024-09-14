# coding: utf-8
# Author: zyx
import math
import os
import shutil
# from tools import detect_landmarks

def getDis(pointX, pointY, lineX1, lineY1, lineX2, lineY2):
    """
    计算点到直线的距离
    Args:
        pointX, pointY: 点
        lineX1, lineY1, lineX2, lineY2:直线的两个端点
    return:
        距离，带正负
    """
    a = lineY2 - lineY1
    b = lineX1 - lineX2
    c = lineX2 * lineY1 - lineX1 * lineY2
    # dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    dis = (a * pointX + b * pointY + c) / (math.pow(a * a + b * b, 0.5))
    return dis


def pose_quality_v1(face_landmarks):
    """
    Args:
        face_landmarks: [[x,y],[],[],[],[]]分别对应[左眼，右眼，鼻子，左嘴角，右嘴角]坐标
    return:
        人脸角度质量分数,0-1,值越大，脸越正,建议阈值0.33
    """
    dis_l = getDis(face_landmarks[2][0], face_landmarks[2][1],
                        face_landmarks[0][0], face_landmarks[0][1],
                        face_landmarks[3][0], face_landmarks[3][1])
    dis_r = getDis(face_landmarks[2][0], face_landmarks[2][1],
                        face_landmarks[1][0], face_landmarks[1][1],
                        face_landmarks[4][0], face_landmarks[4][1])
    # logging.info('pose={}'.format(dis_l / dis_r))
    if math.fabs(dis_l) < 0.001 or math.fabs(dis_r) < 0.001:
        return .0
    if dis_l * dis_r > 0:
        return .0
    score = min(math.fabs(dis_l), math.fabs(dis_r)) / max(math.fabs(dis_l), math.fabs(dis_r))
    if score > 0.33:
        score = score * 2
    if score > 1:
        score = 1.0
    return score

def pose_test():
    # (121,50)(131,52)(129,64)(113,71)(121,72)
    landmarks = [[121,50],[131,52],[129,64],[113,71],[121,72]]
    pose_score = pose_quality_v1(landmarks)
    print(pose_score)

if __name__ == '__main__':
    pose_test()
    root = '/Users/yongxiuzhou/Desktop/人脸质量/测试图片'
    output = '/Users/yongxiuzhou/Desktop/人脸质量/角度测试结果'
    url = "http://172.31.3.133:30717/ai-detector-face/analyze"
    cont = 0
    for img in os.listdir(root):
        if img.startswith('.'):
            continue
        img_path = os.path.join(root, img)
        try:
            landmarks = detect_landmarks(img_path, url)
            pose_score = pose_quality_v1(landmarks)
            output_path = os.path.join(output, '{:.3f}_{}.jpg'.format(pose_score, cont))
            shutil.copy(img_path, output_path)
            cont += 1
            print(cont)
        except:
            pass
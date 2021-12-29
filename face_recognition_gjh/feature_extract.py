import cv2
import dlib
import numpy as np
import pandas as pd
import csv

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

person_name = 'gjh'     # 人名标签
path = "data/gjh.csv"

img = cv2.imread('data/img/gjh.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 灰度化


faces = detector(gray)      # 人脸检测

for face in faces:
    shape = predictor(img, face)        # 68个特征点
    feature = face_reco_model.compute_face_descriptor(img, shape)       # 128维特征向量
    face_feature = np.array(feature,  dtype=object)    # 将其数据结构


    with open(path, "w", newline="") as csvfile:   # w:写， a:添加
        writer = csv.writer(csvfile)        # csv写模块
        face_feature = np.insert(face_feature, 0, person_name, axis=0)  # 打上姓名标签
        print(face_feature)
        writer.writerow(face_feature)   # 写入一行，保存


# cv2.imshow('feature extract',img)
# cv2.waitKey(0)
import dlib
import cv2
import numpy as np
import pandas as pd



# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


feature_list = []           # 已经录入的人脸特征
name_list = []              # 已经录入的人脸名字

current_feature_list = []    # 存储当前摄像头中捕获到的人脸特征

last_name_list = []          # 上一次记录的人脸名字
current_name_list = []       # 存储当前摄像头中捕获到的人脸名字

position = []  # 显示名字区域的坐标

last_face_count = 0         # 上一帧人脸数
current_face_count = 0      # 当前帧人脸数

e_distance_list = [] # 欧式距离

count = 1   # 计数器

# 从 "features_all.csv" 读取录入人脸特征
def get_face_database():
    path = "data/features_all.csv"                  # 人脸128维特征值及其标签数据的路径
    saved_feature = pd.read_csv(path, header=None)   # 人脸特征数据

    for i in range(saved_feature.shape[0]):         # saved_feature.shape[0]为saved_feature.shape的行数
        feature = []
        name_list.append(saved_feature.iloc[i][0])       # 人名

        for j in range(1, 129):
            feature.append(saved_feature.iloc[i][j])

        feature_list.append(feature)

    print("数据库中人脸个数：", len(feature_list))
    return 0

# 计算欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist

# 更新坐标
def refresh_position(faces):
    for i in range(len(faces)):
        # 每个捕获人脸的名字坐标
        position_tem = tuple([faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top()) / 4)])
        position[i] = position_tem

# 人脸识别
def face_recognition(frame, faces, count):
    # 有人脸
    if current_face_count != 0:

        # 人脸检测
        for face in faces:
            # 左上角(x1,y1)，右下角(x2,y2)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # 每隔30帧进行一次人脸识别

        if count >= 30:
            count = 1  # 重置计数器

            if len(faces) > 0:
                print('人脸个数', len(faces))
                refresh_position(faces)  # 更新人脸坐标
                for i in range(len(faces)):
                    shape = predictor(frame, faces[i])  # 功能：定位人脸关键点
                    # 参数：frame：一个numpy ndarray，包含8位灰度或RGB图像
                    # 　　　face：开始内部形状预测的边界框
                    # 返回值：68个关键点的位置

                    current_feature = face_reco_model.compute_face_descriptor(frame,
                                                                              shape)  # 功能：图像中的68个关键点转换为128D面部描述符，其中同一人的图片被映射到彼此附近，并且不同人的图片被远离地映射。
                    # 参数：frame：人脸灰度图，类型：numpy.ndarray
                    # 　　　shape：68个关键点位置
                    # 返回值：128D面部描述符

                    current_feature_list[i] = current_feature  # 当前摄像头中捕获到的人脸特征




                # 欧式距离
                # for i in range(len(faces)):  # 每张人脸
                    e_distance_list = []  # 欧式距离列表

                    for j in range(len(feature_list)):  # 本地人脸特征数
                        # 利用128维特征值计算欧式距离 （当前摄像头中捕获到的人脸特征，已录入的人脸特征）
                        e_distance = return_euclidean_distance(
                            current_feature_list[i],
                            feature_list[j])

                        e_distance_list.append(e_distance)

                    min_e_distant = min(e_distance_list)  # 最小的欧氏距离
                    print(min(e_distance_list))

                    # 找出最小的欧式距离匹配
                    similar_person_index = e_distance_list.index(min_e_distant)  # index() 函数用于从列表中找出某个值第一个匹配项的索引位置

                    if min_e_distant < 0.5:
                        name = name_list[similar_person_index]  # 人脸名
                        current_name_list[i] = name
                        print("第 ", i+1, " 张人脸结果为: ", name_list[similar_person_index])

                    else:
                        current_name_list[i] = 'unknown'
                        print("第 ", i+1, " 张人脸结果为: Unknown person")

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  #更改API设置
flag = cap.isOpened()
get_face_database()


time = 0
while (flag):
    cv2.waitKey(1)  # 延时1毫秒
    count += 1
    time += 1


    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # 水平翻转
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = detector(gray, 0)   # 返回值：人脸检测矩形框4点坐标
                                # 第二个参数越大，代表讲原图放大多少倍在进行检测，提高小人脸的检测效果。

    # 3.  更新人脸数
    last_face_count = current_face_count        # 更新上一帧人脸数
    current_face_count = len(faces)             # 记录本帧人脸数

    # 4. 更新上一帧中的人脸姓名列表 / Update the face name list in last frame
    last_name_list = current_name_list[:]

    # 5.1 如果当前帧和上一帧人脸数没有变化
    if (current_face_count == last_face_count):
        face_recognition(frame, faces, count)       # 人脸识别

    # 5.2 人脸发生变化
    else:
        print("人脸由", last_face_count ,"to", current_face_count)
        current_name_list = []      # 重置当前人名列表
        current_feature_list = []   # 重置当前特征列表
        position = []               # 重置坐标


        if current_face_count == 0:
            print("人脸减少为0")
        else:           # 人脸增加
            """"""
            for face in faces:
                shape = predictor(frame, face)
                current_feature = face_reco_model.compute_face_descriptor(frame, shape)
                current_feature_list.append(current_feature)
                current_name_list.append('unknown')

            # for face in faces:
                # 每个捕获人脸的名字坐标
                position_tem = tuple([face.left(), int(face.bottom() + (face.bottom() - face.top()) / 4)])
                position.append(position_tem)

            """以上为初始化"""



            # face_recognition(frame, faces, count)

    # 将名字显示在幕布上
    for i in range(len(faces)):
        x = position[i][0]
        y = position[i][1]
        s = current_name_list[i]
        frame = cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)  # 图片、文字、位置、字体、字体大小、颜色、粗细

    cv2.imshow("Capture_Paizhao", frame)
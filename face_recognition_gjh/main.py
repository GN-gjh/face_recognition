import dlib
import cv2
import numpy as np
import pandas as pd


# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')            # 功能：标记人脸关键点  参数：‘data/data_dlib/shape_predictor_68_face_landmarks.dat’：68个关键点模型地址  返回值：人脸关键点预测器

# Dlib Resnet 人脸识别模型, 提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")    #功能：生成面部识别器  参数：data/data_dlib/dlib_face_recognition_resnet_model_v1.dat 面部识别模型路径 返回值：面部识别器


class Face_detect:
    def __init__(self):
        self.count = 0
        self.update_interval = 30       # 更新间隔(帧)
        self.feature_list = []          # 已经录入的 人脸特征 列表
        self.current_feature_list = []   # 当前摄像头中的人脸特征 列表
        self.name_list = []             # 已经录入的人脸名字 列表
        self.current_name_list = []     # 当前帧中的人脸姓名 列表
        self.position = []              # 显示姓名区域的坐标
        self.last_face_count = 0        # 上一帧的人脸个数
        self.current_face_count = 0     # 当前帧人脸个数
        self.e_distance_list = []       # 欧式距离 列表


    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self):
        path = "data/features_all.csv"  # 人脸128维特征值及其标签数据的路径
        # path = "gjh.csv"  # 人脸128维特征值及其标签数据的路径
        saved_feature = pd.read_csv(path, header=None)  # 人脸特征数据

        for i in range(saved_feature.shape[0]):  # saved_feature.shape[0]为saved_feature.shape的行数
            feature = []
            self.name_list.append(saved_feature.iloc[i][0])  # 人名

            for j in range(1, 129):
                feature.append(saved_feature.iloc[i][j])

            self.feature_list.append(feature)

        print("数据库中人脸个数：", len(self.feature_list))
        return 0

    # 人脸识别
    def face_recognition(self, frame, faces):
        # 有人脸
        if self.current_face_count != 0:
            # 人脸检测
            for face in faces:
                # 左上角(x1,y1)，右下角(x2,y2)
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 每隔指定更新间隔进行一次人脸识别
            if self.count >= self.update_interval:
                self.count = 1  # 重置计数器

                # 如果存在人脸
                if len(faces) > 0:
                    print('人脸个数', len(faces))
                    for i in range(len(faces)):
                        shape = predictor(frame, faces[i])  # 功能：定位人脸关键点 返回值：68个关键点的位置
                        current_feature = face_reco_model.compute_face_descriptor(frame, shape)  # 功能：图像中的68个关键点转换为128D面部描述符 返回值：128维面部描述符

                        # 更新当前帧中的人脸特征
                        self.current_feature_list[i] = current_feature
                        # 更新当前帧的人脸坐标
                        position_tem = tuple([faces[i].left(), int(faces[i].bottom() + (faces[i].bottom() - faces[i].top()) / 4)])
                        self.position[i] = position_tem

                        """欧式距离"""
                        self.e_distance_list = []  # 欧式距离列表

                        for j in range(len(self.feature_list)):  # 本地人脸特征数
                            # 利用128维特征值计算欧式距离 （当前摄像头中捕获到的人脸特征，已录入的人脸特征）
                            e_distance = self.return_euclidean_distance(self.current_feature_list[i], self.feature_list[j])
                            self.e_distance_list.append(e_distance)

                        min_e_distant = min(self.e_distance_list)  # 最小的欧氏距离
                        print('最小欧式距离：',min(self.e_distance_list))

                        # 找出最小的欧式距离匹配的索引
                        similar_person_index = self.e_distance_list.index(min_e_distant)  # index() 函数用于从列表中找出某个值第一个匹配项的索引位置

                        if min_e_distant < 0.5:
                            name = self.name_list[similar_person_index]  # 人脸名
                            self.current_name_list[i] = name
                            print("第 ", i + 1, " 张人脸结果为: ", self.name_list[similar_person_index])

                        else:
                            self.current_name_list[i] = 'unknown'
                            print("第 ", i + 1, " 张人脸结果为: Unknown person")
                else:
                    print("当前不存在人脸")

    def show_names(self,frame, faces):
        for i in range(len(faces)):
            x = self.position[i][0]
            y = self.position[i][1]
            s = self.current_name_list[i]
            frame = cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)  # 图片、文字、位置、字体、字体大小、颜色、粗细

    # 进程
    def process(self, cap):
        flag = cap.isOpened()       # 摄像头是否打开成功标识
        self.get_face_database()    # 读取录入数据库人脸特征

        while (flag):
            ret, frame = cap.read()     # 参数ret 为True 或者False,代表有没有读取到图片。第二个参数frame表示截取到一帧的图片
            frame = cv2.flip(frame, 1)  # 视频水平翻转
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化

            # 1.获取人脸
            faces = detector(gray, 0)   # 功能：对图像画人脸框  返回值：人脸检测矩形框4点坐标

            # 2.更新人脸数
            self.last_face_count = self.current_face_count  # 更新上一帧人脸数
            self.current_face_count = len(faces)  # 记录本帧人脸数

            # 3.1 当前帧和上一帧人脸数没有变化
            if (self.current_face_count == self.last_face_count):
                self.face_recognition(frame, faces)  # 人脸识别

            # 3.2 人脸发生变化
            else:
                print("人脸由", self.last_face_count, "to", self.current_face_count)
                self.current_name_list = []  # 重置当前人名列表
                self.current_feature_list = []  # 重置当前特征列表
                self.position = []  # 重置坐标

                if self.current_face_count == 0:
                    print("人脸减少为0")
                else:
                    # 人脸增加
                    for face in faces:
                        shape = predictor(frame, face)
                        current_feature = face_reco_model.compute_face_descriptor(frame, shape)
                        self.current_feature_list.append(current_feature)        # 初始化特征列表 并 提取面部特征

                        self.current_name_list.append('unknown')                 # 初始化姓名列表

                        # 初始化坐标数列 并 捕获每个人脸的名字坐标
                        position_tem = tuple([face.left(), int(face.bottom() + (face.bottom() - face.top()) / 4)])
                        self.position.append(position_tem)

            # 4.更新帧
            self.count += 1
            self.show_names(frame, faces)      # 显示人名
            cv2.imshow("Face Detect", frame)  # 显示
            cv2.waitKey(1)  # 延时1毫秒， 每隔1毫秒获取1帧


    def run(self):
        cap = cv2.VideoCapture(0)     # 打开摄像头
        self.process(cap)                           # 进行人脸检测
        
        cap.release()           # 关闭摄像头
        cv2.destroyAllWindows() # 关闭所有窗口


def main():
    face_detect = Face_detect()     # 创建人脸识别实例
    face_detect.run()

if __name__ == '__main__':
    main()




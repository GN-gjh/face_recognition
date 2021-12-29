# face_recognition
bigwork in cqupt university 

### 主要文件
demo.py 是人脸检测基础功能的展示，可以在控制台输出识别出的人脸姓名

feature_extract.py 功能是提取出128维人脸特征矢量，并保存为csv文件

main.py实现人脸识别的全部功能


### data/data_dlib路径下的文件

1.dlib_face_recognition_resnet_model_v1.dat 面部关键点模型 

2.shape_predictor_68_face_landmarks.dat 面部识别模型

需要自行下载，下载地址: http://dlib.net/files/

### 安装依赖库
pip install -r requirements.txt

import dlib
import cv2


# 人脸检测
detector = dlib.get_frontal_face_detector()

# 人脸关键点标注。
predictor = dlib.shape_predictor(
    'data/data_dlib/shape_predictor_68_face_landmarks.dat'
)
img = cv2.imread('data/img/00.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 这样也可以灰度图但是不建议用：im2 = cv2.imread('tfboys.jpg',flags = 0)

faces = detector(gray, 0)  # 第二个参数越大，代表讲原图放大多少倍在进行检测，提高小人脸的检测效果。

for face in faces:
    # 左上角(x1,y1)，右下角(x2,y2)
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    print(x1, y1, x2, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow("image", img)
cv2.waitKey(0)
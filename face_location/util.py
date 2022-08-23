import cv2 as cv
import emotion_distinguish_locate.util
import joblib
import tensorflow as tf
import numpy as np

# 读入多人脸图片
IMAGE_PATH = '../Data/bfd6de4743a6276c5c9d565027f71e5c1660362029024.jpeg'
image_read = cv.imread(IMAGE_PATH)
wrong_num = {6: 4, 5: 2, 4: 6, 3: 3, 2: 5, 1: 0, 0: 1}
LABEL_DICT = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
}

# image_read = cv.resize(image_read, (650, 650))


def get_face(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 调用detectMultiScale函数实现人脸定位
    face_cascade = cv.CascadeClassifier('../opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5),
                                          flags=cv.CASCADE_SCALE_IMAGE)
    return faces


# 切分出人脸图片
# 参数：原图：initial_image,人脸位置大小：faces
# 返回值：人脸图片：image_face
def cut_face_from_image(initial_image, faces):
    image_face = []

    for (x, y, w, h) in faces:
        image_face.append(initial_image[y: y + h, x: x + w, :])
    return image_face


# 表情信息标注
# 参数：使用的模型：model，原图片：image
# 返回值：标注后的图片：image
def emotion_sign(model, image):
    predict_image = []
    faces = get_face(image)
    if len(faces) > 0:
        image_face = cut_face_from_image(image, faces)
        for i in image_face:
            image_p = emotion_distinguish_locate.util.preprocess_read_image(i)
            image_p = emotion_distinguish_locate.util.unify_image(image_p)
            image_p = emotion_distinguish_locate.util.normalize_data(image_p)
            predict_image.append(image_p)
        predict_image = tf.constant(predict_image)
        predict = model.predict(predict_image)
        for i in range(len(predict)):
            cv.putText(image, LABEL_DICT[wrong_num[np.argmax(predict[i])]], (faces[i][0], faces[i][1]), fontFace=0,
                       color=[0, 255, 255],
                       thickness=int(image.shape[0] / 300),
                       fontScale=image.shape[0] / 500)
            cv.rectangle(image, (faces[i][0], faces[i][1]), (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3]),
                         (0, 0, 255), int(image.shape[0] / 600))

    return image


#model = tf.keras.models.load_model('../emotion_distinguish_locate/model/FINAL_CNN_MODEL')
#image = emotion_sign(model, image_read)
#cv.imshow('test', image)
#cv.waitKey()
#cv.destroyAllWindows()

import cv2 as cv
import emotion_distinguish_locate.util
import joblib

# 读入多人脸图片
IMAGE_PATH = '../Data/103884.jpg'
image_read = cv.imread(IMAGE_PATH)
LABEL_DICT = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
}


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
    image_face = cut_face_from_image(image, faces)
    for i in image_face:
        image_p = emotion_distinguish_locate.util.preprocess_read_image(i)
        image_p = emotion_distinguish_locate.util.unify_image(image_p)
        image_p = emotion_distinguish_locate.util.normalize_data(image_p)
        predict_image.append(image_p)

    predict = model.predict(predict_image)
    for i in range(len(predict)):
        cv.putText(image, LABEL_DICT[predict[i]], (faces[i][0], faces[i][1]), fontFace=0, color=[0, 0, 255], thickness=1,
                   fontScale=1)

    return image
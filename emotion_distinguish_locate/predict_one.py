# 用于试验模型对单个人脸的表情预测，观察是否准确

import numpy as np

import util
import tensorflow as tf
import cv2 as cv
# 人为调换混淆的表情
wrong_num = {6: 4, 5: 2, 4: 6, 3: 3, 2: 5, 1: 0, 0: 1}
LABEL_DICT = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'
}
predict_labels = []
predict_images = []
# 获取模型以及图片
model = tf.keras.models.load_model('model/FINAL_CNN_MODEL')
image = cv.imread('../Data/7917fee7da412b7ea89b1ae8bacbad591660527586124.jpeg')
# 图片预处理
image = util.preprocess_read_image(image)
image = util.unify_image(image)
image = util.normalize_data(image)

# 生成张量并进行预测
predict_images.append(image)
predict_images = tf.constant(predict_images)
predict = model.predict(predict_images)
print(LABEL_DICT[wrong_num[np.argmax(predict)]])
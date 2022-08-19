import numpy as np

import util
import tensorflow as tf
import cv2 as cv
wrong_num = {6: 4, 5: 2, 4: 6, 3: 3, 2: 5, 1: 0, 0: 1}
predict_labels = []
predict_images = []
model = tf.keras.models.load_model('model/FINAL_CNN_MODEL')
image = cv.imread('../Data/IMG_20220819_154536.jpg')
image = util.preprocess_read_image(image)
image = util.unify_image(image)
image = util.normalize_data(image)
predict_images.append(image)
predict_images = tf.constant(predict_images)
predict = model.predict(predict_images)
print(wrong_num[np.argmax(predict)])
import cv2 as cv
import util
import numpy as np
import joblib
import tensorflow as tf
predict_labels = []
model = tf.keras.models.load_model('model/cnn_model2')
test_data, test_labels = util.read_images_list('../Data/fer2013_data_strength/test')
test_data = tf.constant(test_data)
test_labels = tf.constant(test_labels)
predict = model.predict(test_data)
for i in predict:
    predict_labels.append(np.argmax(i))
errors = np.count_nonzero(predict_labels - test_labels)
print(errors / len(predict_labels))

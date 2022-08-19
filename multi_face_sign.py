from face_location.util import emotion_sign
import tensorflow as tf
import cv2 as cv
model = tf.keras.models.load_model('emotion_distinguish_locate/model/cnn_model0')
image = cv.imread('Data/103884.jpg')
image = emotion_sign(model, image)
cv.imshow('test', image)
cv.waitKey()
cv.destroyAllWindows()
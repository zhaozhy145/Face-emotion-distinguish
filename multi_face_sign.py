from face_location.util import emotion_sign
import tensorflow as tf
import cv2 as cv


# model = tf.keras.models.load_model('emotion_distinguish_locate/model/FINAL_CNN_MODEL')
# image = cv.imread('Data/_20220823091945.jpg')
# image = cv.resize(image, (570, 300))
# image = emotion_sign(model, image)
# cv.imshow('test', image)
# cv.waitKey()
# cv.destroyAllWindows()

def solute(image):
    model = tf.keras.models.load_model('../emotion_distinguish_locate/model/FINAL_CNN_MODEL')
    image_f = emotion_sign(model, image)
    return image_f

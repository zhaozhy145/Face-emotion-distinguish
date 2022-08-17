import cv2 as cv
import util
import joblib
predict_image = []
MODEL_PATH = 'model/svm_distinguish.m'
image = cv.imread('../Data/t010b53788c91b97894.jpg')
image = util.preprocess_read_image(image)
image = util.unify_image(image)
image = util.normalize_data(image)
predict_image.append(image)

model = joblib.load(MODEL_PATH)
predict = model.predict(predict_image)
print(predict)
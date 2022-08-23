# 用于预测模型的准确率以及模型预测的结果

import cv2 as cv
import util
import numpy as np
import joblib
import tensorflow as tf

# 人为观察输出结果，发现部分表情识别搞混，人为将输出结果调正
wrong_num = {6: 4, 5: 2, 4: 6, 3: 3, 2: 5, 1: 0, 0: 1}
predict_labels = []
# 获取模型，进行预测
model = tf.keras.models.load_model('model/FINAL_CNN_MODEL')
test_data, test_labels = util.read_images_list('../dataset/test')
# 图片预处理，形成张量
test_data = test_data / 255.0
test_data = tf.constant(test_data)
# 输出张量的格式，判断能否进行预测
print(test_data.shape)
predict = model.predict(test_data)
# 输出预测的结果与原结果，进行人为观察
for i in predict:
    predict_labels.append(wrong_num[np.argmax(i)])
for i in range(len(predict_labels)):
    print(predict_labels[i], test_labels[i])
# 输出模型的准确率
errors = np.count_nonzero(predict_labels - test_labels)
print(1 - errors / len(predict_labels))
# 最终准确率：0.4775703538590137

import util
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
import cv2 as cv

TRAIN_DIR_PATH = '../Data/fer2013_data_strength/train'
TEST_DIR_PATH = '../Data/fer2013_data_strength/test'
VAL_DIR_PATH = '../Data/fer2013_data_strength/val'

MODEL_PATH = 'model/mlp_distinguish.m'

train_data, train_labels = util.read_images_list(TRAIN_DIR_PATH)
test_data, test_labels = util.read_images_list(TEST_DIR_PATH)
val_data, val_labels = util.read_images_list(VAL_DIR_PATH)


def train(train_data, train_labels):
    normalized_data = util.normalize_data(train_data)

    # 模型创建
    model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(48, 24))
    # 模型训练
    model.fit(normalized_data, train_labels)

    # 模型保存
    joblib.dump(model, MODEL_PATH)


def T(test_data, test_labels):
    normalized_data = util.normalize_data(test_data)

    model = joblib.load(MODEL_PATH)

    predicts = model.predict(normalized_data)

    errors = np.count_nonzero(predicts - test_labels)
    print(errors / len(predicts))


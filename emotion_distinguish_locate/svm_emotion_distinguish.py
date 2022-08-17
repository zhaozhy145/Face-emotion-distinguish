import numpy as np
import util
import joblib
from sklearn import svm


MODEL_PATH = 'model/svm_distinguish1.m'

# 借助tensorflow使用SVM对表情进行分类识别
# 0:angry,1:disgust,2:fear,3:happy,4:sad,5:surprise,6:neutral
train_images, train_labels = util.read_images_list('../Data/fer2013_data_strength/train')
private_test_images, private_test_labels = util.read_images_list('../Data/fer2013_data_strength/test')

# 3. 定义模型和loss函数
X_train = train_images
y_train = train_labels
model = svm.LinearSVC(max_iter=10)

# 4.开始训练数据
model.fit(X_train, y_train)

# 5.模型保存
joblib.dump(model, MODEL_PATH)

# 6.模型预测
X_test = private_test_images
y_test = private_test_labels
predicts = model.predict(X_test)
errors = np.count_nonzero(predicts - y_test)
print(errors / len(predicts))
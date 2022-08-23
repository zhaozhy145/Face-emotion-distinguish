# 最终确定的模型的训练
import os
import pathlib

import joblib
import tensorflow as tf

# 读取图片路径，生成图片名称集以及标签集
print('开始读取图片路径')
train_root_dir = '../dataset/train'
test_root_dir = '../dataset/test'
root_dir_path = pathlib.Path(train_root_dir)

all_image_filename = [str(jpg_path) for jpg_path in root_dir_path.glob('**/*.jpg')]
all_image_label = [pathlib.Path(image_file).parent.name for image_file in all_image_filename]
all_image_unique_labelname = list(set(all_image_label))
name_index = dict((name, index) for index, name in enumerate(all_image_unique_labelname))
all_image_lable_code = [name_index[pathlib.Path(path).parent.name] for path in all_image_filename]

print('训练路径读取完成')

AUTOTUNE = tf.data.experimental.AUTOTUNE


# 定义map所用的函数
def get_image_by_filename(filename, label):
    print('aa')
    image_data = tf.io.read_file(filename)
    image_jpg = tf.image.decode_jpeg(image_data)
    image_resized = tf.image.resize(image_jpg, [48, 48])
    image_scale = image_resized / 255.0
    print('bb')
    return image_scale, label


# 生成训练集dataset
tf_feature_filenames = tf.constant(all_image_filename)  # X_train
tf_labels = tf.constant(all_image_lable_code)  # y_train
dataset = tf.data.Dataset.from_tensor_slices((tf_feature_filenames, tf_labels))

dataset = dataset.map(map_func=get_image_by_filename, num_parallel_calls=AUTOTUNE)

print('生成cnn网络，完成训练:')
num_epochs = 10  # 训练次数
batch_size = 128  # 训练集分成的每批含有数量
learning_rate = 0.001  # 学习率

# 预处置
print('预处置')
dataset = dataset.shuffle(buffer_size=200000)
dataset = dataset.batch(batch_size=batch_size)
dataset = dataset.prefetch(AUTOTUNE)

# 模型定义
model = tf.keras.models.Sequential()
# 第一段
# 第一卷积层，64个大小为5×5的卷积核，步长1，激活函数relu，卷积模式same，输入张量的大小
model.add(tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 第一池化层，池化核大小为2×2，步长2
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))  # 随机丢弃40%的网络连接，防止过拟合
# 第二段
model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))
# 第三段
model.add(tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(tf.keras.layers.Flatten())  # 过渡层
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2048, activation='relu'))  # 全连接层
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))  # 分类输出层
# 模型训练
model.summary()
model.build()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
)
model.fit(dataset, epochs=num_epochs)
# model.evaluate()
# 模型保存
model.save('model/cnn_model9')

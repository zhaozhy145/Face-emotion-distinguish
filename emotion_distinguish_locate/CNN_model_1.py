import os
import pathlib
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
import tensorflow as tf

MODEL_PATH = 'saves/cnn_distinguish.m'

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


def get_image_by_filename(filename, label):
    print('aa')
    image_data = tf.io.read_file(filename)
    image_jpg = tf.image.decode_jpeg(image_data)
    image_resized = tf.image.resize(image_jpg, [48, 48])
    image_scale = image_resized / 255.0
    print('bb')
    return image_scale, label


tf_feature_filenames = tf.constant(all_image_filename)  # X_train
tf_labels = tf.constant(all_image_lable_code)  # y_train
dataset = tf.data.Dataset.from_tensor_slices((tf_feature_filenames, tf_labels))

dataset = dataset.map(map_func=get_image_by_filename, num_parallel_calls=AUTOTUNE)

print('生成cnn网络，完成训练:')
num_epochs = 10  # 训练次数
batch_size = 128  # 训练集分成的每批含有数量
learning_rate = 0.01  # 学习率

print('预处置')
dataset = dataset.shuffle(buffer_size=200000)
dataset = dataset.batch(batch_size=batch_size)
dataset = dataset.prefetch(AUTOTUNE)

model = tf.keras.Sequential([
    Conv2D(32, (1, 1), strides=(1, 1), input_shape=(48, 48, 1), padding='same', activation='relu',
           kernel_initializer='uniform'),
    Conv2D(32, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'),
    MaxPooling2D(pool_size=(2, 2)),

    Dense(2048, activation='relu'),
    Dropout(0.2),

    Dense(1024, activation='relu'),
    Dropout(0.2),

    Dense(7, activation='softmax'),
])
model.summary()
model.build()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
)
model.fit(dataset, epochs=num_epochs)
# model.evaluate()
'''
几次运行结果：
loss: 0.3049 - sparse_categorical_crossentropy: 0.3049
'''
model.save('model/cnn_model13')

import os
import pathlib

import joblib
import tensorflow as tf

# 读取图片路径，生成图片名称集以及标签集
print('开始读取图片路径')
root_dir = '../Data/fer2013_data_strength/train'
root_dir_path = pathlib.Path(root_dir)

all_image_filename = [str(jpg_path) for jpg_path in root_dir_path.glob('**/*.jpg')]
all_image_label = [pathlib.Path(image_file).parent.name for image_file in all_image_filename]
all_image_unique_labelname = list(set(all_image_label))
name_index = dict((name, index) for index, name in enumerate(all_image_unique_labelname))
all_image_lable_code = [name_index[pathlib.Path(path).parent.name] for path in all_image_filename]

print('路径读取完成')

AUTOTUNE = tf.data.experimental.AUTOTUNE


# 定义map所用的函数
def get_image_by_filename(filename, label):
    print('aa')
    image_data = tf.io.read_file(filename)
    image_jpg = tf.image.decode_jpeg(image_data)
    image_resized = tf.image.resize(image_jpg, [48, 48])
    image_scale = image_resized / 47.0
    print('bb')
    return image_scale, label


# 生成训练集dataset
tf_train_feature_filenames = tf.constant(all_image_filename)  # X_train
tf_train_labels = tf.constant(all_image_lable_code)  # y_train
train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_feature_filenames, tf_train_labels))

train_dataset = train_dataset.map(map_func=get_image_by_filename, num_parallel_calls=AUTOTUNE)

print('生成cnn网络，完成训练:')
num_epochs = 10  # 训练次数
batch_size = 1000  # 训练集分成的每批含有数量
learning_rate = 0.001  # 学习率

# 预处置
print('预处置')
train_dataset = train_dataset.shuffle(buffer_size=200000)
train_dataset = train_dataset.batch(batch_size=batch_size)
train_dataset = train_dataset.prefetch(AUTOTUNE)

# 模型定义
model = tf.keras.Sequential([  # 卷积,池化,卷积,池化,卷积,全连接,relu,softmax
    tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(16, 5, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(120, 5, activation='relu'),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')
])
# 模型训练
model.build()
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
)
model.fit(train_dataset, epochs=num_epochs)
# 模型保存
model.save('model/cnn_model0')

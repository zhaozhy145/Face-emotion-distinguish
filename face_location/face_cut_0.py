import pathlib
import re
import linecache
import os
import cv2 as cv
import tensorflow as tf
from PIL import Image

# 本程序用于将wider face数据集中label部分分离出来并且重新保存
import numpy as np

FILEDIR = '../Data'
file = open('../Data/wider_face_split/wider_face_train_bbx_gt.txt')


def count_lines(file):
    lines_quantity = 0
    while True:
        buffer = file.read(1024 * 8192)
        if not buffer:
            break
        lines_quantity += buffer.count('\n')
    file.close()
    return lines_quantity


lines = count_lines(file)


def cut_image(FILEDIR):
    face_image = []
    face_labels = []
    roi = []

    for i in range(lines):
        line = linecache.getline('../Data/wider_face_split/wider_face_train_bbx_gt.txt', i)
        if re.search('jpg', line):
            position = line.index('/')
            file_name = line[position + 1: -5]
            folder_name = line[:position]
            i += 1
            face_count = int(linecache.getline(FILEDIR + os.sep + 'wider_face_split/wider_face_train_bbx_gt.txt', i))
            for j in range(face_count):
                box_line = linecache.getline(FILEDIR + os.sep + 'wider_face_split/wider_face_train_bbx_gt.txt',
                                             i + j + 1)  # x1, y1, w, h, x1,y1 为人脸框左上角的坐标
                po_x1 = box_line.index(' ')
                x1 = box_line[:po_x1]
                po_y1 = box_line.index(' ', po_x1 + 1)
                y1 = box_line[po_x1:po_y1]
                po_w = box_line.index(' ', po_y1 + 1)
                w = box_line[po_y1:po_w]
                po_h = box_line.index(' ', po_w + 1)
                h = box_line[po_w:po_h]
                po_b = box_line.index(' ', po_h + 1)
                b = box_line[po_h:po_b]
                coordinates = x1 + y1 + w + h
                if not (os.path.exists("../Data/WIDER_train/images/" + folder_name)):
                    os.makedirs("../Data/WIDER_train/image/" + folder_name)
                path = pathlib.Path("../Data/WIDER_train/images/" + folder_name + "/" + file_name + ".jpg")
                image = cv.imread(str(path))
                roi = image[int(x1):int(x1) + int(w):, int(y1):int(y1) + int(h)]  # 截取人脸部分
                image_resize = cv.resize(image, (227, 227))
                face_image.append(image[int(y1): int(y1) + int(h), int(x1): int(x1) + int(w), :].ravel())
                face_labels.append(int(b))
            i += i + j + 1
    return roi, np.array(face_image), np.array(face_labels)


# 导入目录下的所有jpg文件
# 自动计算并行优化的条件数量
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_image_by_filename(roi, label):
    # 借助 tf 读入文件
    image_jpg = tf.image.decode_jpeg(roi)
    image_resized = tf.image.resize(image_jpg, [48, 48])
    image_scale = image_resized / 255.0
    return image_scale, label


def get_datasets(roi, label):
    # 图片特征
    tf_train_feature = tf.constant(roi)
    # 图片标签
    tf_train_labels = tf.constant(label)
    # 生成dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((tf_train_feature, tf_train_labels))
    train_dataset = train_dataset.map(map_func=get_image_by_filename, num_parallel_calls=AUTOTUNE)
    return train_dataset


def model_train(train_dataset):
    # 3. 生成cnn网络，完成训练
    num_epochs = 15
    batch_size = 16
    learning_rate = 0.001

    # 3-0: 为了做训练，对 datset 执行一些预处置
    train_dataset = train_dataset.shuffle(buffer_size=200000)
    train_dataset = train_dataset.batch(batch_size=batch_size)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    # 3-1： 使用 tf.keras 的序贯模型完成cnn网络结构的声明
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(120, 5, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='sigmod'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_crossentropy]
    )

    return model.fit(train_dataset, epochs=num_epochs)


if __name__ == '__main__':
    roi, _, labels = cut_image(FILEDIR)
    train_dataset = get_datasets(roi, labels)
    model_train(train_dataset)

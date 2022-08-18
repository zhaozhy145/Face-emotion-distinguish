import cv2 as cv
import os
import numpy as np

LABEL_DICT = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6
}


# 图片读入并形成列表
# 参数：图片根目录：image_dir_path
# 返回值：图片列表：data, 标签列表：labels
def read_images_list(image_dir_path):
    data = []
    labels = []

    for item in os.listdir(image_dir_path):  # 获取根目录下的文件目录
        item_path = os.path.join(image_dir_path, item)  # 生成通向标签的文件路径
        if os.path.isdir(item_path):  # 判断是否为路径
            for subitem in os.listdir(item_path):  # 获取特征下的文件目录（图片名称）
                subitem_path = os.path.join(item_path, subitem)  # 生成通向图片的文件路径
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)  # 读入灰度图
                data.append(gray_image)  # 将图片展平添加到图片数据集
                labels.append(LABEL_DICT[item])  # 将标签添加到标签集

    return np.array(data), np.array(labels)


# 对读入的图片进行预处理
# 参数：图片：image_read
# 预处理后的图片：preprocess_image
def preprocess_read_image(image_read):
    # 高斯模糊
    blured_image = cv.GaussianBlur(image_read, (5, 5), 0)
    # 转成灰度图
    preprocess_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)

    return preprocess_image


# 统一尺寸：48×48
# 参数：需要统一尺寸的人脸图片：plate_image
# 返回值：统一尺寸后的人脸图片：uniformed_image
def unify_image(plate_image):
    # 声明统一的尺寸
    PLATE_STD_HEIGHT = 48
    PLATE_STD_WIDTH = 48
    # 完成resize
    uniformed_image = cv.resize(plate_image, (PLATE_STD_WIDTH, PLATE_STD_HEIGHT))
    return uniformed_image


# 标准化
# 参数：特征矩阵：data
# 返回值：执行标准化后的data
def normalize_data(data):
    return (data - data.mean()) // data.max()

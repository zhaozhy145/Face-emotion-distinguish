import re
import linecache
import os
import cv2 as cv

# 本程序用于将wider face数据集中label部分分离出来并且重新保存
import numpy as np

FILEDIR = "../Data/"
file = open(FILEDIR + 'wider_face_split/wider_face_train_bbx_gt.txt', 'r')


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

    for i in range(lines):
        line = linecache.getline(FILEDIR + 'wider_face_split/wider_face_train_bbx_gt.txt', i)
        if re.search('jpg', line):
            position = line.index('/')
            file_name = line[position + 1: -5]
            folder_name = line[:position]
            i += 1
            face_count = int(linecache.getline(FILEDIR + 'wider_face_split/wider_face_train_bbx_gt.txt', i))
            for j in range(face_count):
                box_line = linecache.getline(FILEDIR + 'wider_face_split/wider_face_train_bbx_gt.txt',
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
                if not (os.path.exists(FILEDIR + "WIDER_train/image/" + folder_name)):
                    os.makedirs(FILEDIR + "WIDER_train/image/" + folder_name)
                image = cv.imread(FILEDIR + "WIDER_train/images/" + folder_name + "/" + file_name + ".jpg")
                face_image.append(image[int(y1): int(y1) + int(h), int(x1): int(x1) + int(w), :].ravel())
                face_labels.append(int(b))
            i += i + j + 1
    return np.array(face_image), np.array(face_labels)

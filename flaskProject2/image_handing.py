import cv2 as cv
import numpy as np


def image_handing(image_readed):
    rows, cols = len(image_readed), len(image_readed[0])
    for i in range(rows):
        for j in range(cols):
            for pixel in range(3):
                image_readed[i][j][pixel] = 255 - image_readed[i][j][pixel]

    # cv.imshow('title', image_readed)
    # cv.waitKey()
    # cv.destroyAllWindows()
    return image_readed

# if __name__ == '__main__':
#     image_path = 'static/images/2022080518152400_c.jpg'
#     image_readed = cv.imread(image_path)
#     new_image = image_handing(image_readed)



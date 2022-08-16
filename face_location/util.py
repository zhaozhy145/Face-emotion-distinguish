import cv2 as cv

# 读入多人脸图片
IMAGE_PATH = '../Data/103884.jpg'
image_read = cv.imread(IMAGE_PATH)

cv.imshow('test', image_read)
cv.waitKey()
cv.destroyAllWindows()

gray = cv.cvtColor(image_read, cv.COLOR_BGR2GRAY)

# 调用detectMultiScale函数实现人脸定位
face_cascade = cv.CascadeClassifier('../opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5),
                                      flags=cv.CASCADE_SCALE_IMAGE)
print("发现{0}个人脸!".format(len(faces)))

for (x, y, w, h) in faces:
    cv.circle(image_read, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 255, 0), 2)

cv.imshow("Find Faces!", image_read)
cv.waitKey(0)
cv.destroyAllWindows()


# 切分出人脸图片
# 参数：原图：initial_image,人脸位置大小：faces
# 返回值：人脸图片：image_face
def cut_face_from_image(initial_image, faces):
    image_face = []

    for (x, y, w, h) in faces:
        image_face.append(initial_image[y: y + h, x: x + w, :])
    return image_face


image_face = cut_face_from_image(image_read, faces)
for i in image_face:
    cv.imshow('test', i)
    cv.waitKey()
cv.destroyAllWindows()

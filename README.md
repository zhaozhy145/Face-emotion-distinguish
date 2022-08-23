# Face-emotion-distinguish
12组企业实训最后项目

emotion_distinguish_locate文件夹里面放的是用于表情识别训练的文件，主要包含工具类文件util，以及三种模型的训练文件，包括mlp、SVM、CNN，加上一个对模型结果进行预测的文件：predict，用于对模型的准确率进行预测，加上一个predict_one，用于对单张人脸进行识别
util文件里包含对数据集的读取的read_images_list，对图片预处理的preprocess_read_image、unify_image、normalize_data

face_location文件夹里放着人脸定位的工具类util，用于人脸定位的模型，我们使用了OpenCV自带的haarcascades中的haarcascade_frontalface_default.xml模型
util文件里包括用于调用haarcascade_frontalface_default.xml模型，获取人脸坐标及大小的get_face，切分人脸并形成数据集的cut_face_from_image，在图片上对每张人脸进行表情预测并在人脸位置进行标注的emotion_sign

flaskProject2文件夹里是用于可视化的文件

使用时，可以运行flaskProject2文件夹下的app.py文件，此时会生成一个网站，我们可以通过网站选择我们需要进行表情识别的图片并点击上传，我们就可以得到一张标注人脸以及对应表情的图片

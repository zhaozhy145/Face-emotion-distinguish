# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, flash
from werkzeug.utils import secure_filename
import os
import cv2
import time
import flask

from datetime import timedelta
from image_handing import image_handing
from multi_face_sign import solute

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return render_template('upload.html')
        user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images',
                                   secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'preview.jpg'), img)

        img_1 = solute(img)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'result.jpg'), img_1)

        return render_template('upload_ok.html', userinput=user_input, val1=time.time())

    return render_template('upload.html')


@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':

        upload_path_video = "F:/Dev/WorkSpace/PyCharm/Face_emotion_distinguish/Data/image.png"
        img = cv2.imread(upload_path_video)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), 'static/images', 'image.jpg'), img)

        img_new = solute(img)
        cv2.imwrite(os.path.join(os.path.dirname(__file__), 'static/images', 'image_new.jpg'), img_new)
        return render_template('video_ok.html')
    else:
        return render_template('new.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8987, debug=True)

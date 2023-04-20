from flask import Flask, render_template, request

from YOLO5.Model import YOLO

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def image_addres():
    if request.method == 'POST':
        try:
            YOLO(request.form['content'], r'YOLO5\configuration_files\yolov5s.onnx',
                 r'YOLO5\configuration_files\classes.txt')
            return 'result of processing saved in image folder'
        except Exception as error:
            return str(error)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)

import cv2
import matplotlib
import numpy as np
import sys
import tensorflow as tf
from distutils.version import StrictVersion
from googletrans import Translator
import os
from flask import Flask, render_template
from werkzeug.utils import secure_filename
from flask import request

app = Flask(__name__)
sys.path.append("..")

def find_tag(dict):
    if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
      raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')
    from utils import label_map_util
    matplotlib.use('tkagg')

    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    PATH_TO_IMAGE = os.path.join(dict)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

      sess = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_expanded})

    objects = []
    threshold = 0.00

    for index, value in enumerate(classes[0]):
        object_dict = {}
        if scores[0, index] > threshold:
            object_dict["score"] = \
                scores[0, index]
            object_dict["name"] = \
                (category_index.get(value)).get('name')
            objects.append(object_dict)

    rate = 0
    category = ''
    for object in objects:
        if object["score"] > rate:
            rate = object["score"]
            category = object["name"]

    return category

class Result:
    tr = ""
    fr = ""
    de = ""
    es = ""
    en = ""


def translate(tag):
    result = Result()
    translator = Translator()
    result.en = tag
    result.tr = translator.translate(tag, dest='tr').text
    result.fr = translator.translate(tag, dest='fr').text
    result.de = translator.translate(tag, dest='de').text
    result.es = translator.translate(tag, dest='es').text
    return result

@app.route('/')
def upload_file1():
    return render_template('upload.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    f = request.files['file']
    f.save("pictures/" + secure_filename(f.filename))
    dict = "pictures/" + secure_filename(f.filename)
    tag = find_tag(dict)
    if tag != "":
        html = "<html><head><title>BK Visual Dictionary</title></head><body>"
        result = translate(tag)
        body = "<table align='center'><tr><td><b>English:</b> " + result.en + "</td></tr><tr><td> <b>Spanish:</b> "+ result.es + "</td></tr><tr><td> <b>German:</b> " + result.de + "</td></tr><tr><td> <b>French:</b> " + result.fr + "</td></tr><tr><td> <b>Turkish:</b> " + result.tr + "</td></tr></table>"
        html = html + body + "</body>"
        return html
    else:
        return "No object detected"

if __name__ == '__main__':
    app.run()



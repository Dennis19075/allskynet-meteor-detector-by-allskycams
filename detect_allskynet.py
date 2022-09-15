from curses import flash
from doctest import debug
from io import BytesIO
import mimetypes
from PIL import Image
from flask import Flask
from flask import render_template, url_for
from flask import request, redirect, flash
from flask import Response
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

from werkzeug.utils import secure_filename

import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

app = Flask(__name__)

def __init__(self):
    upload_image()
      

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "meteor"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 640 * 640

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

CUSTOM_MODEL_NAME = 'allskynet' 
# PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'IMAGE_PATH': os.path.join('static'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join(CUSTOM_MODEL_NAME, 'export', 'checkpoint'),
 }

files = {
    'PIPELINE_CONFIG':os.path.join(CUSTOM_MODEL_NAME, 'export', 'pipeline.config'),
    'LABELMAP': os.path.join(LABEL_MAP_NAME)
}


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore the best checkpoint or frozen checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
# CAM_PATH = 'rtsp://192.168.76.74/user=admin&password=&channel=1&stream=0.sdp'

def generate(filename):
        IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], filename)
        image_np = load_image_into_numpy_array(IMAGE_PATH)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1 #label class id 1 for meteor
        image_np_with_detections = image_np.copy()
        # Printing the 5 best detection scores
        print(detections['detection_scores'][0])
        print(detections['detection_scores'][1])
        print(detections['detection_scores'][2])
        print(detections['detection_scores'][3])
        print(detections['detection_scores'][4])
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)
        (flag, encodedImage) = cv2.imencode(".jpg", image_np_with_detections)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html") #image_path it is just a value to get the images to the html

@app.route("/image_feed/<filename>")
def image_feed(filename):
    return Response(generate(filename),
        mimetype="multipart/x-mixed-replace; boundary=frame")

# load the image
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return render_template(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        print(filename)
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)
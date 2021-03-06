import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import feature_extractor_wrapper
import threading

from utils import label_map_util

from utils import visualization_utils as vis_util

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as im

import cv2

cap = cv2.VideoCapture(1)
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


fps = cap.get(cv2.CAP_PROP_FPS)
feature_extractor = feature_extractor_wrapper.feature_extractor_wrapper("ft_ResNet50", 8)
feature_extractor.load_reID()

# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:


# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1

# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if 'frozen_inference_graph.pb' in file_name:
#        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

# In[6]:

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:

def load_image_into_numpy_array(image):
    im_width, im_height, channels = image.shape
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[9]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# In[10]:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        t_first = time.time_ns()
        imgArray = []
        t1 = None
        while True:
            # Start time
            #print("Picture took ", (time.time_ns() - t_first)/1000000000, " seconds")
            t_first = time.time_ns()
            start = time.time()
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            # Filter only class persons
            indices = np.argwhere(classes == 1)
            boxes = np.squeeze(boxes[indices])
            scores = np.squeeze(scores[indices])
            classes = np.squeeze(classes[indices])


            # Denormalize Coordinates so we can crop the image
            # Get Image shape
            img_height, img_width, img_channel = image_np.shape
            absolute_coord = []
            THRESHOLD = 0.7  # adjust your threshold here

            # Get length of boxes
            boxlen = len(boxes)
            for i in range(boxlen):
                if scores[i] < THRESHOLD:
                    continue
                box = boxes[i]
                ymin, xmin, ymax, xmax = box
                x_up = int(xmin * img_width)
                y_up = int(ymin * img_height)
                x_down = int(xmax * img_width)
                y_down = int(ymax * img_height)
                absolute_coord.append((x_up, y_up, x_down, y_down))

            # actually cropping
            bounding_box_img = []
            
            i = 0
            for c in absolute_coord:
                bounding_box_img.append(image_np[c[1]:c[3], c[0]:c[2], :])
                #cv2.imwrite("./test/abc" + str(i) + ".jpg", bounding_box_img[i])
                im_np = bounding_box_img[i]
                #im_pil = im.fromarray(cv2.cvtColor(bounding_box_img[i], cv2.COLOR_BGR2RGB))
                #t1 = threading.Thread(target=feature_extractor_wrapper.start_reID, args=(fe, rb, [im_pil]))
                #t1.start()
                imgArray.append(im_np)
                if len(imgArray) == 8:
                    t_join_start = time.time_ns()
                    if t1 is not None:
                        t1.join()
                    t_join_end = time.time_ns()
                    #print("Waited for Thread for", (t_join_end - t_join_start)/1000000000, "seconds")
                    #feature_extractor.start_reID(imgArray)
                    t1 = threading.Thread(target=feature_extractor.start_reID, args=(imgArray,))
                    t1.start()
                    imgArray = []

                # im_pil.save("./test/pill" + str(i) + ".jpg")
                i += 1
            
            #vis_util.visualize_boxes_and_labels_on_image_array(
            #    image_np,
            #    np.squeeze(boxes),
            #    np.squeeze(classes).astype(np.int32),
            #    np.squeeze(scores),
            #    category_index,
            #    use_normalized_coordinates=True,
            #    line_thickness=8)
            
            
            # End time
            end = time.time()
            #print("endtime : {0}".format(end))
            # Time elapsed
            seconds = end - start
            
            fps  = 1/seconds
            #print ("Estimated frames per second : {0}".format(fps))
            #print ("Test : {0}".format(cap.get(1)))
            #print(load_image_into_numpy_array(image_np))
            #cv2.imshow('np', load_image_into_numpy_array(image_np))
            #cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            #cv2.imshow('im_np', im_np)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()
            #    break

import numpy as np
import os
import tensorflow as tf
import time
from utils import label_map_util
import feature_extractor_wrapper

import threading
# from utils import visualization_utils as vis_util
# from PIL import Image

import cv2

THRESHOLD = 0.7

# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
#MODEL_NAME = 'rfcn_resnet101_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1
CONF = 0
CNT_FRAMES = 0
RES_FPS = 0
ALL_DET = 0
BATCH_SIZE = 8
img_array = []
t1 = None


# functions


def filter_classes(cl, bx, sc):
    indices = np.argwhere(cl == 1)
    fboxes = np.squeeze(bx[indices])
    fscores = np.squeeze(sc[indices])
    fclasses = np.squeeze(cl[indices])
    return fboxes, fscores, fclasses


def denormalize_coordinates(img, bx, sc):
    result = []
    img_height, img_width, img_channel = img.shape
    # absolute_coord = []
    boxlen = len(bx)
    for i in range(boxlen):
        if sc[i] < THRESHOLD:
            continue
        box = bx[i]
        ymin, xmin, ymax, xmax = box
        x_up = int(xmin * img_width)
        y_up = int(ymin * img_height)
        x_down = int(xmax * img_width)
        y_down = int(ymax * img_height)
        result.append((x_up, y_up, x_down, y_down))
    return result


def cropping_roi(absarr, img):
    i = 0
    result = []
    for c in absarr:
        result.append(img[c[1]:c[3], c[0]:c[2], :])
        # cv2.imwrite("./test/abc" + str(i) + ".jpg", result[i])
        # im_pil = Image.fromarray(cv2.cvtColor(bounding_box_img[i], cv2.COLOR_BGR2RGB))
        # im_pil.save("./test/pill" + str(i) + ".jpg")
        i += 1

    return result


def detect(img):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(img, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    dboxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    dscores = detection_graph.get_tensor_by_name('detection_scores:0')
    dclasses = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (rdboxes, rdscores, rdclasses, rdnum_detections) = sess.run(
        [dboxes, dscores, dclasses, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return rdboxes, rdscores, rdclasses


(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("test4.mp4")

feature_extractor = feature_extractor_wrapper.feature_extractor_wrapper("ft_ResNet50", BATCH_SIZE)
feature_extractor.load_reID()

# ## Load a (frozen) Tensorflow model into memory.

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal
# utility functions, but anything that returns a dictionary mapping integers to appropriate
# string labels would be fine


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection:

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            start = time.time()
            ret, image_np = cap.read()



            startDetect = time.time()
            # detect
            boxes, scores, classes = detect(image_np)

            detectTime = time.time() - startDetect

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            # Filter only class persons
            boxes, scores, classes = filter_classes(classes, boxes, scores)

            # Denormalize Coordinates so we can crop the image
            absolute_coord = denormalize_coordinates(image_np, boxes, scores)

            # actually cropping
            bounding_box_imgs = cropping_roi(absolute_coord, image_np)

            for img in bounding_box_imgs:
                img_array.append(img)


            if len(img_array) >= BATCH_SIZE:
                if t1 is not None:
                    t1.join()

                t1 = threading.Thread(target=feature_extractor.start_reID, args=(img_array,))
                t1.start()

                img_array = []
            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #    image_np,
            #    boxes,
            #    classes.astype(np.int32),
            #    scores,
            #    category_index,
            #    use_normalized_coordinates=True,
            #    line_thickness=8)

            # cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
            #
            # endtime = time.time()
            # seconds = endtime - start
            # fps = (1 / seconds)

            # print("--------------------------------\n")
            # print("Frame: " + str(CNT_FRAMES))
            # print("Estimated frames per second : {0}".format(fps))
            # print("Detection took: " + str(detectTime))

            # for score in scores:
            #     if score > 0:
            #         CONF += score
            #         print("Score is: " + str(score))
            # print("\n")
            # CNT_FRAMES += 1
            # RES_FPS += fps
            # AVGFPS = int(RES_FPS) / int(CNT_FRAMES)
            # ALL_DET += detectTime
            # AVGDECTTIME = float(ALL_DET) / float(CNT_FRAMES)
            # AVGCONF = float(CONF) / float(CNT_FRAMES)

            # print("AVGFPS: " + str(AVGFPS))
            # print("AVGDECTTIME: " + str(AVGDECTTIME))
            # print("AVGCONF: " + str(AVGCONF))

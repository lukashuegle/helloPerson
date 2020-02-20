from feature_extractor_class import feature_extraction
from ringbuffer import Ringbuffer
import logging
import time
import numpy as np
from PIL import Image
import cv2
import threading
import sayhello

THRESHHOLD_RE_ID = 0.65
THRESHHOLD_NEW_ID = 1.05
FEATURE_SIZE = 60

class feature_extractor_wrapper:

    def __init__(self, model, batch_size):
        self.last_image_shown = time.time() - 1000
        self.model = model
        self.batch_size = batch_size
        self.threading_queue = []
        self.sayhello = sayhello.Sayhello(1, 150)
        image_show_thread = threading.Thread(target=self.image_viewer)
        image_show_thread.start()


    def load_reID(self):
        logging.basicConfig(filename='feature_extractor_wrapper.log', level=logging.DEBUG)
        self.feature_extractor = feature_extraction('0', self.model, '1', self.batch_size)
        self.ringbuffer = Ringbuffer(200, 10000)
        #return feature_extractor, ringbuffer

    def start_reID(self, img_array):
        t_singleimg_start = time.time_ns()
        feature_array = self.feature_extractor.extract_feature(img_array)
        t_singleimg_end = time.time_ns()
        logging.debug("Batch feature extraction took" + str((t_singleimg_end - t_singleimg_start)/1000000000) + "seconds")
        #print(feature)
        count = 0
        for feature in feature_array:
            feature = [feature]
            #print(feature)
            smallest_index, smallest_distance = self.ringbuffer.nearestneighbors(feature)
            print(smallest_distance)
            logging.debug("Smallest distance: " + str(smallest_distance))
            print(len(self.ringbuffer.ringbuffer))
            logging.debug("Length of ringbuffer: " + str(len(self.ringbuffer.ringbuffer)))
            if self.ringbuffer.ringbuffer:
                logging.debug("Length of ringbuffer[0]" + str(len(self.ringbuffer.ringbuffer[0])))
            if smallest_distance <= THRESHHOLD_RE_ID:
                last_seen = self.ringbuffer.lastseen(smallest_index)
                self.sayhello.sayagain_async(last_seen)
                self.ringbuffer.addnewfeature(smallest_index, feature)
                img_old = self.ringbuffer.getimage(smallest_index)
                #if (time.time() - self.last_image_shown) >= 5:
                self.update(img_old)
                
            elif smallest_distance >= THRESHHOLD_NEW_ID:
                self.ringbuffer.addnewperson(feature, np.array(img_array[count]))
                self.sayhello.sayhello_async()
                

            count += 1

    def update(self, image):
        self.last_image_shown = time.time()
        self.threading_queue.append(image)

    def image_viewer(self):
        img = np.array(Image.open("../../testdir/WIN_20200218_11_23_35_Pro (2).jpg"))
        while True:
            if len(self.threading_queue) > 0:
                img = self.threading_queue.pop(0)
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            cv2.waitKey(30)
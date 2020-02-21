from feature_extractor_class import feature_extraction
from ringbuffer import Ringbuffer
import logging
import time
import numpy as np
from PIL import Image
import cv2
import threading
import sayhello
import datetime
import collections

THRESHHOLD_RE_ID = 0.65
THRESHHOLD_NEW_ID = 1.059
FEATURE_SIZE = 60

class feature_extractor_wrapper:

    def __init__(self, model, batch_size):
        self.last_image_shown = time.time() - 1000
        self.model = model
        self.batch_size = batch_size
        self.threading_queue = []
        self.speaking_queue = collections.deque(maxlen=1)
        self.sayhello = sayhello.Sayhello(1, 150)
        self.last_person = -1
        self.last_seen_person = time.time() - 10
        image_show_thread = threading.Thread(target=self.image_viewer)
        text_speak_thread = threading.Thread(target=self.text_to_speech_t)
        text_speak_thread.start()
        image_show_thread.start()


    def load_reID(self):
        logging.basicConfig(filename='feature_extractor_wrapper.log', level=logging.DEBUG)
        self.feature_extractor = feature_extraction('0', self.model, '1', self.batch_size)
        self.ringbuffer = Ringbuffer(200, 10000)
        #return feature_extractor, ringbuffer

    def start_reID(self, img_array):
        t_singleimg_start = time.time_ns()
        feature_array = self.feature_extractor.extract_feature_numpy(img_array)
        t_singleimg_end = time.time_ns()
        logging.debug("Batch feature extraction took" + str((t_singleimg_end - t_singleimg_start)/1000000000) + "seconds")
        #print(feature)
        count = 0
        for feature in feature_array:
            feature = [feature]
            #print(feature)
            smallest_index, smallest_distance = self.ringbuffer.nearestneighbors(feature)
            #print(smallest_distance)
            logging.debug("Smallest distance: " + str(smallest_distance))
            #print(len(self.ringbuffer.ringbuffer))
            logging.debug("Length of ringbuffer: " + str(len(self.ringbuffer.ringbuffer)))
            if self.ringbuffer.ringbuffer:
                logging.debug("Length of ringbuffer[0]" + str(len(self.ringbuffer.ringbuffer[0])))
            if smallest_distance <= THRESHHOLD_RE_ID:
                print("Erkannt")
                last_seen, person_id = self.ringbuffer.lastseen(smallest_index)
                #self.sayhello.sayagain_async(last_seen)
                self.ringbuffer.addnewfeature(smallest_index, feature)
                img_old = self.ringbuffer.getimage(smallest_index)
                if person_id != self.last_person:
                    print("Hallo ich habe sie das letzte mal", last_seen, "gesehen")
                    self.last_person = person_id
                    self.speaking_queue.append(last_seen)
                elif (time.time() - self.last_seen_person) > 5:
                    print("Hallo ich habe sie das letzte mal", last_seen, "gesehen")
                    self.last_person = person_id
                    self.speaking_queue.append(last_seen)
                else:
                    print(time.time() - self.last_seen_person)
                #self.last_person = person_id
                self.update(img_old, person_id)
                self.last_seen_person = time.time()
                
            elif smallest_distance >= THRESHHOLD_NEW_ID:
                self.ringbuffer.addnewperson(feature, np.array(img_array[count]))
                print("Herzlich Willkommen!")
                self.speaking_queue.append(1)
                #self.sayhello.sayhello_async()
                

            count += 1

    def update(self, image, person_id):
        self.last_image_shown = time.time()
        self.threading_queue.append([image, person_id],)

    def image_viewer(self):
        img = cv2.cvtColor(np.array(Image.open("../../testdir/initial.png")), cv2.COLOR_RGB2BGR)
        while True:
            if len(self.threading_queue) > 0:
                img_info = self.threading_queue.pop(0)
                img = img_info[0]
                if self.last_person != img_info[1]:
                    cv2.destroyAllWindows()
            cv2.imshow("image", img)
            cv2.waitKey(30)

    def text_to_speech_t(self):
        #last_spoken = time.time() - 5
        obj = None
        #time.sleep(0.1)
        while True:
            time.sleep(0.000000000001)
            #print(len(self.speaking_queue))
            #print((time.time() - last_spoken))
            if len(self.speaking_queue) > 0:
                obj = self.speaking_queue.pop()
            #self.last_person = -1
            if obj is not None:
                if obj == 1:
                    self.sayhello.sayhello()
                    obj = None
                    if len(self.speaking_queue) > 0:
                        self.speaking_queue.pop()
                    #last_spoken = time.time()
                else:
                    self.sayhello.sayagain(obj)
                    obj = None
                    if len(self.speaking_queue) > 0:
                        self.speaking_queue.pop()
                    #last_spoken = time.time()

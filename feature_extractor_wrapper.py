from feature_extractor_class import feature_extraction
from ringbuffer import Ringbuffer
import logging
import time

THRESHHOLD_RE_ID = 0.6
THRESHHOLD_NEW_ID = 1.05
FEATURE_SIZE = 60

def load_reID(model, batch_size):
    logging.basicConfig(filename='feature_extractor_wrapper.log', level=logging.DEBUG)
    feature_extractor = feature_extraction('0', model, '1', batch_size)
    ringbuffer = Ringbuffer(200, 10000)
    return feature_extractor, ringbuffer

def start_reID(feature_extractor, ringbuffer, img_array):

    t_singleimg_start = time.time_ns()
    feature_array = feature_extractor.extract_feature(img_array)
    t_singleimg_end = time.time_ns()
    logging.debug("Batch feature extraction took" + str((t_singleimg_end - t_singleimg_start)/1000000000) + "seconds")
    #print(feature)
    for feature in feature_array:
        feature = [feature]
        #print(feature)
        smallest_index, smallest_distance = ringbuffer.nearestneighbors(feature)
        #print(smallest_distance)
        logging.debug("Smallest distance: " + str(smallest_distance))
        #print(len(ringbuffer.ringbuffer))
        logging.debug("Length of ringbuffer: " + str(len(ringbuffer.ringbuffer)))
        if ringbuffer.ringbuffer:
            logging.debug("Length of ringbuffer[0]" + str(len(ringbuffer.ringbuffer[0])))
        if smallest_distance <= THRESHHOLD_RE_ID:
            ringbuffer.addnewfeature(smallest_index, feature)
            last_seen = ringbuffer.lastseen(smallest_index)
            print("Letztes mal gesehen: " + last_seen)
            
        elif smallest_distance >= THRESHHOLD_NEW_ID:
            ringbuffer.addnewperson(feature)
            print("Willkommen!")

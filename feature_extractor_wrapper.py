from feature_extractor_class import feature_extraction

THRESHHOLD = 0.5
FEATURE_SIZE = 60

def load_reID(model):
    feature_extractor = feature_extraction('0', model, '1')
    ringbuffer = ringbuffer(FEATURE_SIZE, 200, 100)
    return feature_extractor

def start_reID(feature_extractor, ringbuffer, img_array):

    for img in img_array:
        feature = feature_extractor.extract_feature([img])[0]
        smallest_index, smallest_distance = ringbuffer.nearestneighbors(feature)
        if smallest_distance <= THRESHHOLD:
            ringbuffer.addnewfeature(smallest_index, feature)
            
        else:
            ringbuffer.addnewperson([feature])

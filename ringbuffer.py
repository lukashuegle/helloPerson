from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import collections
import numpy as np
import logging


class Ringbuffer:
    def __init__(self, personnumber, featurenumber):
        self.ringbuffer = collections.deque(maxlen=personnumber)
        self.timebuffer = collections.deque(maxlen=personnumber)
        self.imagebuffer = collections.deque(maxlen=personnumber)
        self.featurenumber = featurenumber
        self.id = 0
        logging.basicConfig(filename='ringbuffer_class.log', level=logging.DEBUG)

    def addnewperson(self, featurearray, image):
        self.ringbuffer.append(featurearray)
        self.timebuffer.append([datetime.now(), self.id])
        self.id = self.id + 1
        self.imagebuffer.append(image)

    def addnewfeature(self, position, feature):
        if len(self.ringbuffer[position]) >= self.featurenumber:
            self.ringbuffer[position] = self.ringbuffer[position][1:self.featurenumber]
        self.ringbuffer[position] = np.concatenate((np.array(self.ringbuffer[position]), np.array(feature)), axis=0)
        # idold = self.timebuffer[position][1]
        self.timebuffer[position][0] = datetime.now()

    def getimage(self, position):
        return self.imagebuffer[position]

    def lastseen(self, position):
        return self.timebuffer[position][0], self.timebuffer[position][1]

    def nearestneighbors(self, newefeature):
        smallestdistance = 100
        indexofsmallest = -1
        if self.ringbuffer:
            distance = np.arange(0)
            for person in self.ringbuffer:
                neigh = NearestNeighbors(n_neighbors=1, algorithm='brute')
                neigh.fit(person)
                distances, _ = neigh.kneighbors(newefeature)
                distance = np.append(distance, np.amin(distances))
                smallestdistance = np.amin(distance)
                indexofsmallest = (np.where(distance == smallestdistance))[0][0]
        else:
            logging.debug("Ringbuffer is empty")
        return indexofsmallest, smallestdistance

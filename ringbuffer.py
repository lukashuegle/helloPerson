from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import collections
import numpy as np
import logging


class Ringbuffer:
    def __init__(self, personnumber, featurenumber):
        self.ringbuffer = collections.deque(maxlen=personnumber)
        self.timebuffer = collections.deque(maxlen=personnumber)
        self.featurenumber = featurenumber
        logging.basicConfig(filename='ringbuffer_class.log', level=logging.DEBUG)

    def addnewperson(self, featurearray):
        self.ringbuffer.append(featurearray)
        self.timebuffer.append(datetime.now())

    def addnewfeature(self, position, feature):
        if len(self.ringbuffer[position]) >= self.featurenumber:
            self.ringbuffer[position] = self.ringbuffer[position][1:self.featurenumber]
        self.ringbuffer[position] = np.concatenate((np.array(self.ringbuffer[position]), np.array(feature)), axis=0)
        self.timebuffer[position] = datetime.now()

    def lastseen(self, position):
        return self.timebuffer[position]

    def nearestneighbors(self, newefeature):
        smallestdistance = 100
        indexofsmallest = -1
        if self.ringbuffer:
            distance = np.arange(0)
            for person in self.ringbuffer:
                neigh = NearestNeighbors(n_neighbors=1)
                neigh.fit(person)
                distances, _ = neigh.kneighbors(newefeature)
                distance = np.append(distance, np.amin(distances))
                smallestdistance = np.amin(distance)
                indexofsmallest = (np.where(distance == smallestdistance))[0][0]
        else:
            logging.debug("Ringbuffer is empty")
        return indexofsmallest, smallestdistance


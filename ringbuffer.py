from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import collections
import numpy as np


class Ringbuffer:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    ringbuffer = collections.deque(maxlen=1000)
    timebuffer = collections.deque(maxlen=1000)
    now = datetime.now()

    def addnewperson(self, featurearray):
        self.ringbuffer.append([featurearray])
        self.timebuffer.append(self.now.strftime("%H:%M:%S"))

    def addnewfeature(self, position, feature):
        self.ringbuffer[position] = np.concatenate((np.array(self.ringbuffer[position]), np.array([feature])), axis=0)
        self.timebuffer[position] = self.now.strftime("%H:%M:%S")

    def nearestneibors(self, newefeature):
        distance = np.arange(0)
        for person in self.ringbuffer:
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(person)
            distances, _ = neigh.kneighbors(newefeature)
            distance = np.append(distance, np.amin(distances))
        smallestdistance = np.amin(distance)
        indexofsmallest = (np.where(distance == smallestdistance))[0][0]
        return indexofsmallest, smallestdistance

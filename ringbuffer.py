from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import collections
import numpy as np


class Ringbuffer:
    def __init__(self, personnumber, featurenumber):
        self.ringbuffer = collections.deque(maxlen=personnumber)
        self.timebuffer = collections.deque(maxlen=personnumber)
        self.featurenumber = featurenumber

    def addnewperson(self, featurearray):
        self.ringbuffer.append(featurearray)
        self.timebuffer.append(datetime.now().strftime("%H:%M:%S"))

    def addnewfeature(self, position, feature):
        if len(self.ringbuffer[position])>= self.featurenumber:
            print("ich bin hier drin")
            self.ringbuffer[position] = self.ringbuffer[position][1:self.featurenumber]
        self.ringbuffer[position] = np.concatenate((np.array(self.ringbuffer[position]), np.array([feature])), axis=0)
        self.timebuffer[position] = datetime.now().strftime("%H:%M:%S")

    def nearestneighbors(self, newefeature):
        distance = np.arange(0)
        for person in self.ringbuffer:
            neigh = NearestNeighbors(n_neighbors=1)
            print(self.ringbuffer)
            neigh.fit(person)
            distances, _ = neigh.kneighbors(newefeature)
            distance = np.append(distance, np.amin(distances))
        smallestdistance = 100
        indexofsmallest = -1
        if distance.size > 0:
            smallestdistance = np.amin(distance)
            indexofsmallest = (np.where(distance == smallestdistance))[0][0]
        return indexofsmallest, smallestdistance




import numpy as np
import cv2 as cv
import imutils
import time
from imutils.object_detection import non_max_suppression
import logging

logging.basicConfig(filename='HOGDetection.log', level=logging.DEBUG)

# initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
font = cv.FONT_HERSHEY_SIMPLEX
cap = cv.VideoCapture("test.mp4")
#cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

counter = 0
while True:
    # Capture frame-by-frame
    startFrame = time.time()
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    logging.debug('----------------------')
    logging.debug('')
    logging.debug('FRAME:' + str(counter))
    
    # resizing for faster detection
    frame = cv.resize(frame, (640, 480))
    
    start = time.time()
    orig = frame.copy()
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8))
    # draw the original bounding boxes
    end = time.time()
    logging.debug('detection:' + str((end - start)))
    #print("[INFO] HOG took {:6f} seconds".format(end - start))
    
    #for x, y, w, h in rects:
    #    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    i = 0
    for xA, yA, xB, yB in rects:
        logging.debug('confidences:' + str(weights[i]))
        #print("[INFO] HOG took {" + str(weights[i]) + "} confidences")
        #cv.putText(frame, 'Test:' + str(weights[i]), (10,450), font, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
        i+=1
    
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    # draw the final bounding boxes
    #for xA, yA, xB, yB in pick:
    #    cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    endFrame = time.time()
    seconds = endFrame - startFrame
    logging.debug('fps:' + str((1/seconds)))
    logging.debug('')
    #print("[INFO] HOG took {:6f} fps".format((1/(endFrame - startFrame))))
    
    # show the output images
    #cv.imshow("Before NMS", orig)
    cv.imshow("After NMS", frame)

    key_hit = cv.waitKey(33)
    if key_hit == ord('q') or key_hit == 27:  # Esc or q key to stop
        break
    counter+=1
cap.release()    
cv.destroyAllWindows()



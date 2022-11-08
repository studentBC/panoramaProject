import cv2
import math
import sys
import os
import numpy as np
global width, height, frame, fps

def extractImages(pathIn):
    success,img = cap.read()
    count = 0
    while success:
        cv2.imwrite("./tmp/frame%d.jpg" % count, img)     # save frame as JPEG file      
        #image.append(img)
        success,img = cap.read()
        #print('Read a new frame: ', success)
        count += 1
        if count == 10000:
            break
    if count != frame:
        print("error !!! for ", count)

# read file from mp4 and parse W, L
filePath = sys.argv[1]
cap = cv2.VideoCapture(filePath)
os.mkdir("./tmp")
if cap.isOpened():
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps =  cap.get(cv2.CAP_PROP_FPS)
    print("we have "+str(width) + " "+ str(height)+" "+ str(frame) + " " + str(fps))
    extractImages(filePath)


# remove foreground and create panorama


# display your foreground objects as a video sequence against a white plain background frame by frame.

# Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.

# save video


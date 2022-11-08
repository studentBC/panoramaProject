import cv2
import math
import sys
import os
import argparse
import time
import shutil
import numpy as np
global width, height, frameCount, fps, foreGround, backGround

#we extract foreground and background
#https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
#https://pythonprogramming.net/mog-background-reduction-python-opencv-tutorial/
def extractImages():
    for i in range(frameCount):
        success,img = cap.read()
        if not success:
            print("error in extracting foreground !!!")
        #cv2.imwrite("./tmp/frame%d.jpg" % count, img)     # save frame as JPEG file    
        fgmask = fgbg.apply(img)  
        cv2.imshow('frame',fgmask)
        if i > 10000:
            cv2.imwrite("./tmp/frame%d.jpg" % i, img)
            #foreGround.append(img)
        elif i > 10050:
            break
        #image.append(img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
if __name__=="__main__":
    # read file from mp4 and parse W, L
    filePath = sys.argv[1]
    cap = cv2.VideoCapture(filePath)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    shutil.rmtree("./tmp")
    os.mkdir("./tmp")
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps =  cap.get(cv2.CAP_PROP_FPS)
        print("we have "+str(width) + " "+ str(height)+" "+ str(frameCount) + " " + str(fps))
        extractImages()


    # remove foreground and create panorama


    # display your foreground objects as a video sequence against a white plain background frame by frame.

    # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.

    # save video
    cv2.destroyAllWindoes()
    cap.release()
import cv2
import math
import sys
import os
import argparse
import time
import shutil
import imutils
from StitchPanorama import StitchPanorama
import numpy as np
width = 0
height = 0
frameCount = 0 
fps = 0 
foreGround, backGround, fgmasks = [], [], []


def mse(img1, img2):
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(height*width))
    return mse
#using BFS to get foreground and background
def fillBackground():
    Threshold = 10
    B, G, R, count = 0, 0 ,0, 0
    for a in range(0, frameCount):
        #cv2.imwrite("./tmp/frame%d.jpg" % a, backGround[a])
        for b in range(0, frameCount):
            #if the missing pixel is found in other frame then we can fill the missing one
            m = mse(backGround[a], backGround[b])
            if a != b and m < Threshold:
                for i in range(0, height):
                    for j in range(0, width):
                        #fgmask = 0 means background
                        sa = backGround[a][i][j][0]+backGround[a][i][j][1]+backGround[a][i][j][2]
                        sb = backGround[b][i][j][0]+backGround[b][i][j][1]+backGround[b][i][j][2]
                        if fgmasks[a][i][j] and fgmasks[b][i][j] == 0 and sa == 0 and sb:
                            backGround[a][i][j][0] = backGround[b][i][j][0]
                            backGround[a][i][j][1] = backGround[b][i][j][1]
                            backGround[a][i][j][2] = backGround[b][i][j][2]
                            fgmasks[a][i][j] = 0
                        elif fgmasks[a][i][j] == 0 and fgmasks[b][i][j] and sb == 0 and sa:
                            backGround[b][i][j][0] = backGround[a][i][j][0]
                            backGround[b][i][j][1] = backGround[a][i][j][1]
                            backGround[b][i][j][2] = backGround[a][i][j][2]
                            fgmasks[b][i][j] = 0
        for i in range(0, height):
            for j in range(0, width):
                #if the pixel is still 0 then we use average of its surrounding pixel to fill it
                if fgmasks[a][i][j]:
                    I = i-1
                    while I > -1 and fgmasks[a][I][j]:
                        I-=1
                    if I > -1:
                        B+=backGround[a][I][j][0]
                        G+=backGround[a][I][j][1]
                        R+=backGround[a][I][j][2]
                        count+=1
                    I = i+1
                    while I < height and fgmasks[a][I][j]:
                        I+=1
                    if I < height:
                        B+=backGround[a][I][j][0]
                        G+=backGround[a][I][j][1]
                        R+=backGround[a][I][j][2]
                        count+=1
                    J = j-1
                    while J > -1 and fgmasks[a][i][J]:
                        J-=1
                    if J > -1:
                        B+=backGround[a][i][J][0]
                        G+=backGround[a][i][J][1]
                        R+=backGround[a][i][J][2]
                        count+=1
                    J = j+1
                    while J < width and fgmasks[a][i][J]:
                        J+=1
                    if J < width:
                        B+=backGround[a][i][J][0]
                        G+=backGround[a][i][J][1]
                        R+=backGround[a][i][J][2]
                        count+=1

                    if count:
                        backGround[a][i][j][0] = B/count
                        backGround[a][i][j][1] = G/count
                        backGround[a][i][j][2] = R/count
                    else:
                        print("maybe go diagnoal direction ???")
        cv2.imwrite("./tmp/frame%d.jpg" % a, backGround[a])
        



#we extract foreground and background
#https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
#https://pythonprogramming.net/mog-background-reduction-python-opencv-tutorial/
def extractImages():
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (300,120,470,350)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC()
    for a in range(frameCount):
        success,img = cap.read()
        if not success:
            print("error in extracting foreground !!!")
        #cv2.imwrite("./tmp/frame%d.jpg" % count, img)     # save frame as JPEG file    
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(img)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) 
        # print(fgmask)
        # for i in range(len(fgmask)):
        #     for j in range(len(fgmask[0])):
        #         print(fgmask[i][j], end = ", ")
        #     print()
        #cv2.grabCut(img,fgmask, rect,  bgdModel, fgdModel,  5, cv2.GC_INIT_WITH_RECT)
        #bg = fgbg.getBackgroundImage(img)
        #cv2.imshow('fg mask',fgmask)
        #cv2.imshow('frame', bg)
        #cv2.imwrite("./tmp/frame%d.jpg" % i, fgmask)
        #print(len(img), len(img[0]), len(img[0][0]), img[0][0][0])
        fg, bg = np.zeros((height, width, 3)), np.zeros((height, width, 3))# dont know why it does not work
        fg = fg.astype('uint8')
        bg = bg.astype('uint8')
        prev = 0
        for i in range(0, height):
            for j in range(0, width):
                #print(fgmask[i][j], end = " ")
                if fgmask[i][j] == 0 and prev == 0:
                    bg[i][j][0] = img[i][j][0]
                    bg[i][j][1] = img[i][j][1]
                    bg[i][j][2] = img[i][j][2]
                else:
                    fg[i][j][0] = img[i][j][0]
                    fg[i][j][1] = img[i][j][1]
                    fg[i][j][2] = img[i][j][2]
                prev = fgmask[i][j]
                #print(fg[i][j], img[i][j])
                #print()
        #sometimes the color that cv2 img show is not correct and not realtime
        #when this happen just save processed image to verify
        #cv2.imshow('frame', img)
        #cv2.imshow('fg ',np.asarray(fg))
        #cv2.imshow('bg ', fg)
        #cv2.imshow('frame', bg)
        #cv2.imwrite("./tmp/frame%d.jpg" % a, bg)
        foreGround.append(fg)
        backGround.append(bg)
        fgmasks.append(fgmask)
        #image.append(img)

        # another model which uses BFS
        # pass the frame to the background subtractor
        # foreground_mask = background_subtr_method.apply(img)
        # # obtain the background without foreground mask
        # background_img = background_subtr_method.getBackgroundImage()
        #cv2.imshow('lol', foreground_mask)
        #cv2.imwrite("./tmp/frame%d.jpg" % i, background_img)
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps =  cap.get(cv2.CAP_PROP_FPS)
        frameCount = 10
        print("we have "+str(width) + " "+ str(height)+" "+ str(frameCount) + " " + str(fps))
        extractImages()


    # remove foreground and fill out the removed part in background
    # this issue involved camera motion, size change, object tracking
    fillBackground()
    # using processed background and stitch them together to create panorama
    # https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/ 
    # stitchy = cv2.Stitcher.create()
    # stitchy = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    # ret, panorama = stitchy.stitch(backGround)
    # if ret != cv2.STITCHER_OK:
    #     print("error occur in stitching error code: ", ret)
    sp = StitchPanorama()
    for i in range(1, frameCount):
        sp.stitch(backGround[i-1], backGround[i])

    # display your foreground objects as a video sequence against a white plain background frame by frame.
    
    # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.

    # save video
    
    cap.release()
    cv2.destroyAllWindows()

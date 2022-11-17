import cv2
import math
import sys
import os
import argparse
import time
import shutil
import numpy as np

from foreground_extraction import ForegroundExtractor
from StitchPanorama import StitchPanorama
from tqdm import tqdm

FG_GRABCUT = "grabcut"
FG_MOG = "mog"
FG_MOG2 = "mog2"
FG_GSOC = "gsoc"
FG_GMG = "gmg"
FG_HOG = "hog"
FG_DOF = "dof"
FG_LKO = "lko"
FG_MV = "mv" #motion vector

panoramas = []


def mse(img1, img2):
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    height, width, _ = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(height*width))
    return mse
#using BFS to get foreground and background
def fillBackground(bg, fgmasks):
    print("Filling background...")
    frame_count, height, width, channel = bg.shape
    Threshold = 10
    B, G, R, count = 0, 0 ,0, 0
    for a in tqdm(range(frame_count)):
        #cv2.imwrite("./tmp/frame%d.jpg" % a, backGround[a])
        for b in range(frame_count):
            #if the missing pixel is found in other frame then we can fill the missing one
            m = mse(bg[a], bg[b])
            if a != b and m < Threshold:
                for i in range(height):
                    for j in range(width):
                        #fgmask = 0 means background
                        sa = bg[a][i][j][0]+bg[a][i][j][1]+bg[a][i][j][2]
                        sb = bg[b][i][j][0]+bg[b][i][j][1]+bg[b][i][j][2]
                        if fgmasks[a][i][j] and fgmasks[b][i][j] == 0 and sa == 0 and sb:
                            bg[a][i][j][0] = bg[b][i][j][0]
                            bg[a][i][j][1] = bg[b][i][j][1]
                            bg[a][i][j][2] = bg[b][i][j][2]
                            fgmasks[a][i][j] = 0
                        elif fgmasks[a][i][j] == 0 and fgmasks[b][i][j] and sb == 0 and sa:
                            bg[b][i][j][0] = bg[a][i][j][0]
                            bg[b][i][j][1] = bg[a][i][j][1]
                            bg[b][i][j][2] = bg[a][i][j][2]
                            fgmasks[b][i][j] = 0
        for i in range(height):
            for j in range(width):
                #if the pixel is still 0 then we use average of its surrounding pixel to fill it
                if fgmasks[a][i][j]:
                    I = i-1
                    while I > -1 and fgmasks[a][I][j]:
                        I-=1
                    if I > -1:
                        B+=bg[a][I][j][0]
                        G+=bg[a][I][j][1]
                        R+=bg[a][I][j][2]
                        count+=1
                    I = i+1
                    while I < height and fgmasks[a][I][j]:
                        I+=1
                    if I < height:
                        B+=bg[a][I][j][0]
                        G+=bg[a][I][j][1]
                        R+=bg[a][I][j][2]
                        count+=1
                    J = j-1
                    while J > -1 and fgmasks[a][i][J]:
                        J-=1
                    if J > -1:
                        B+=bg[a][i][J][0]
                        G+=bg[a][i][J][1]
                        R+=bg[a][i][J][2]
                        count+=1
                    J = j+1
                    while J < width and fgmasks[a][i][J]:
                        J+=1
                    if J < width:
                        B+=bg[a][i][J][0]
                        G+=bg[a][i][J][1]
                        R+=bg[a][i][J][2]
                        count+=1

                    if count:
                        bg[a][i][j][0] = B/count
                        bg[a][i][j][1] = G/count
                        bg[a][i][j][2] = R/count
                    else:
                        print("maybe go diagnoal direction ???")
        cv2.imwrite("./tmp/frame%d.jpg" % a, bg[a])

def extract_foreground(frames, args):
    print("Extracting foreground...")
    fgmasks = []
    extractor = ForegroundExtractor()
    if args.fgmode == FG_GRABCUT:
        fgmasks = extractor.get_foreground_mask_grabcut(frames)
    elif args.fgmode == FG_MOG:
        fgmasks = extractor.get_foreground_mask_mog(frames)
    elif args.fgmode == FG_MOG2:
        fgmasks = extractor.get_foreground_mask_mog2(frames)
    elif args.fgmode == FG_GSOC:
        fgmasks = extractor.get_foreground_mask_gsoc(frames)
    elif args.fgmode == FG_GMG:
        fgmasks = extractor.get_foreground_mask_gmg(frames)
    elif args.fgmode == FG_HOG:
        fgmasks = extractor.get_foreground_mask_hog(frames)
    elif args.fgmode == FG_DOF:
        fgmasks = extractor.get_foreground_mask_dof(frames)
    elif args.fgmode == FG_MV:
        fgmasks = extractor.get_foreground_mask_mv(frames, int(args.mv_blocksize), int(args.mv_k), float(args.mv_threshold))
    else:
        print("Invalid fgmode!")
        sys.exit(-1)

    bgmasks = np.where((fgmasks == 1), 0, 1).astype('uint8')
    fg = frames * fgmasks[:, :, :, np.newaxis]
    bg = frames * bgmasks[:, :, :, np.newaxis]
    return fg, bg, fgmasks

def showVideo(frames, fps, filename):
    for i in range(frames.shape[0]):
        cv2.imshow(filename, frames[i])
        if cv2.waitKey(1000 // fps) & 0xFF == ord('q'):
            break

def get_frames(cap):
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    return np.array(frames)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("-fg", "--fgmode", default=FG_GSOC)
    parser.add_argument("-bs","--mv_blocksize", default=16)
    parser.add_argument("-k","--mv_k", default=16)
    parser.add_argument("-th","--mv_threshold", default=15)
    return parser.parse_args()

def main(args):
    cap = cv2.VideoCapture(args.filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps =  int(cap.get(cv2.CAP_PROP_FPS))

    frames = get_frames(cap)
    fg, bg, fgmasks = extract_foreground(frames, args)
    # remove foreground and fill out the removed part in background
    # this issue involved camera motion, size change, object tracking
    fillBackground(bg, fgmasks)
    # using processed background and stitch them together to create panorama
    # we just need to sample 5 points for stitching Q1 - Q5
    test = [bg[0],  bg[5], bg[10]]
    sp = StitchPanorama(test)
    cv2.imwrite("simplePanorama.jpg", sp.simpleStitch())
    cv2.imwrite("ola.jpg", sp.getPanorama())
    # pp = bg[1]
    # fianlFrame = []
    # # since the very beginning frame are usually black we just skip it but in fact
    # # we should search for black one and determine which frame is start frame i just too lazy...
    # panoramas.append(bg[0])
    # panoramas.append(bg[1])
    # for i in range(2, frame_count):
    #     rev, nextp = sp.stitch(pp, bg[i])
    #     panoramas.append(nextp)
    #     pp = nextp
    # cv2.imwrite("panorama.jpg", pp)
    # display your foreground objects as a video sequence against a white plain background frame by frame.
    # https://www.etutorialspoint.com/index.php/319-python-opencv-overlaying-or-blending-two-images
    for i in range(frame_count):
        #print(len(foreGround[i]), len(foreGround[i][0]))
        #print(len(panoramas[i]), len(panoramas[i][0]))
        new_h, new_w, channels = panoramas[i].shape
        resize = cv2.resize(fg[i], (new_w, new_h))
        dst = cv2.addWeighted(resize, 0.5, panoramas[i], 0.7, 0)
        fianlFrame.append(dst)

    # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.
    # save video
    video=cv2.VideoWriter('result.mp4', -1,fps,(width,height))
    #sv = cv2.VideoWriter('./tmp/result.mp4', -1, fps, (height, width))
    for f in fianlFrame:
        video.write(f)

    video.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    args = parse_args()
    main(args)

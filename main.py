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
from matcher import matcher
from fill_background import FillBackGround

FG_GRABCUT = "grabcut"
FG_MOG = "mog"
FG_MOG2 = "mog2"
FG_GSOC = "gsoc"
FG_GMG = "gmg"
FG_HOG = "hog"
FG_DOF = "dof"
FG_LKO = "lko"
FG_MV = "mv" #motion vector
FG_DST = "dst"
panoramas = []

                    



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
    elif args.fgmode == FG_DST:
        fgmasks = extractor.get_foreground_mask_dst(frames)
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
    # fillBackground(bg, fgmasks)
    fbg = FillBackGround()
    fbg.fill_background(bg, fgmasks, fps)
    # using processed background and stitch them together to create panorama
    # we just need to sample 5 points for stitching Q1 - Q5
    sampleBG = [ frames[i] for i in range(0, frame_count, 10) ]
    sp = StitchPanorama(sampleBG)
    pp = sp.simpleStitch()
    cv2.imwrite("simplePanorama.jpg", pp)
    match = matcher()
    fianlFrame = []
    resize = (fg[0].shape[1], fg[0].shape[0])
    #cv2.imwrite("ola.jpg", sp.getPanorama())
    # display your foreground objects as a video sequence against a white plain background frame by frame.
    # https://www.etutorialspoint.com/index.php/319-python-opencv-overlaying-or-blending-two-images
    for a in tqdm(range(frame_count)):
        #print(len(foreGround[i]), len(foreGround[i][0]))
        #print(len(panoramas[i]), len(panoramas[i][0]))
        p = pp 
        h = match.match(pp, fg[a], "lol")
        # print('------- h is --------')
        # print(h)
        # print('---------------------')
        if h is None:
            continue
        for i in range(fg[0].shape[0]):
            for j in range(fg[0].shape[1]):
                if fgmasks[a][i][j]:
                    pos = np.dot(h, [i, j, 1])
                    if pos[0] < 0:
                        pos[0] = -pos[0]
                    elif pos[1] < 0:
                        pos[1] = -pos[1]
                    pos[0] = min(max(0, pos[0]), p.shape[0]-1)
                    pos[1] = min(max(0, pos[1]), p.shape[1]-1)
                    # print(pos)
                    p[int(pos[0])][int(pos[1])][0] = fg[a][i][j][0]
                    p[int(pos[0])][int(pos[1])][1] = fg[a][i][j][1]
                    p[int(pos[0])][int(pos[1])][2] = fg[a][i][j][2]
        # new_h, new_w, channels = panoramas[i].shape
        # resize = cv2.resize(fg[i], (new_w, new_h))
        # dst = cv2.addWeighted(resize, 0.5, p, 0.7, 0)
        cv2.imshow('lol', p)
        k = cv2.waitKey(30) & 0xff
        fianlFrame.append(p)

    # # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.
    # # save video
    video=cv2.VideoWriter('result.mp4', -1, fps,(pp.shape[1],pp.shape[0]))
    #sv = cv2.VideoWriter('./tmp/result.mp4', -1, fps, (height, width))
    for f in fianlFrame:
        video.write(f)

    video.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    args = parse_args()
    main(args)

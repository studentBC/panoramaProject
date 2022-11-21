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
from fillBackground import fillBackGround

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
    # fillBackground(bg, fgmasks)

    fbg = fillBackGround()
    #whether we want to use processed image as panorama 
    use = False 
     # using processed background and stitch them together to create panorama
    # we just need to sample 5 points for stitching Q1 - Q5
    if use: sampleBG = fbg.fill_background(bg, fgmasks, fps)
    else: sampleBG = [ frames[i] for i in range(0, frame_count, 10) ]
    sp = StitchPanorama(sampleBG)
    panoramas = sp.simpleStitch()
    cv2.imwrite("simplePanorama.jpg", panoramas)
    #we detect human and mark it as background
    print(len(panoramas), len(panoramas[0]), len(panoramas[0][0]))
    if not use:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(panoramas, winStride=(8,8) )
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        print(len(panoramas),  len(panoramas[0]), len(panoramas[0][0]))
        for (xA, yA, xB, yB) in boxes:
            print(xA, xB, yA, yB)
            for i in range(yA, yB):
                for j in range(xA, xB):
                    panoramas[i][j][0] = 255
                    panoramas[i][j][1] = 255
                    panoramas[i][j][2] = 255
            # panoramas[yA:yB][xA:xB][0] = 0
            # panoramas[yA:yB][xA:xB][1] = 0
            # panoramas[yA:yB][xA:xB][2] = 0
        cv2.imwrite("processedSimplePanorama.jpg", panoramas)
    #cv2.imwrite("ola.jpg", sp.getPanorama())
    # display your foreground objects as a video sequence against a white plain background frame by frame.
    # https://www.etutorialspoint.com/index.php/319-python-opencv-overlaying-or-blending-two-images
    # for i in range(frame_count):
    #     #print(len(foreGround[i]), len(foreGround[i][0]))
    #     #print(len(panoramas[i]), len(panoramas[i][0]))
    #     new_h, new_w, channels = panoramas[i].shape
    #     resize = cv2.resize(fg[i], (new_w, new_h))
    #     dst = cv2.addWeighted(resize, 0.5, panoramas[i], 0.7, 0)
    #     fianlFrame.append(dst)
    fianlFrame = []
    for f in fg:
        cv2.imwrite('tmp.jpg', fg)
        overlay = cv2.imread('tmp.jpg', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
        h, w = overlay.shape[:2]

        background = panoramas
        for y in range(h):
            for x in range(w):
                overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
                overlay_alpha = overlay[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0

                # get the color from the background image
                background_color = background[y, x]

                # combine the background color and the overlay color weighted by alpha
                composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha

                # update the background image in place
                background[y, x] = composite_color
        fianlFrame.append(background)

    # # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.
    # # save video
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

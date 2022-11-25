import cv2
import math
import sys
import os
import argparse
import time
import shutil
import numpy as np

from libs.StitchPanorama import StitchPanorama
from libs.fill_background import FillBackGround
from tqdm import tqdm
from libs.matcher import matcher
from libs.video import Video

panoramas = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("-fg", "--fgmode", default=Video.FG_MOG2)
    parser.add_argument("-bs", "--mv_blocksize", default=16)
    parser.add_argument("-k", "--mv_k", default=16)
    parser.add_argument("-th", "--mv_threshold", default=15)
    return parser.parse_args()


def main(config: argparse.Namespace) -> None:
    with Video(config.filepath) as cap:
        fg, bg, fgmasks = cap.extract_foreground(config.fgmode, config)
        # remove foreground and fill out the removed part in background
        # this issue involved camera motion, size change, object tracking
        # fillBackground(bg, fgmasks)
        fbg = FillBackGround()
        sampleBG = fbg.fill_background(bg, fgmasks, cap.fps)

        # using processed background and stitch them together to create panorama
        # we just need to sample 5 points for stitching Q1 - Q5
        # sampleBG = [ bg[i] f or i in range(0, frame_count, fps) ]
        sp = StitchPanorama(sampleBG)
        cv2.imwrite("simplePanorama.jpg", sp.simpleStitch())

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)

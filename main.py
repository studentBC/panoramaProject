import argparse
import math
import os
import glob
import shutil
import sys
import time

import cv2
import numpy as np
from tqdm import tqdm

from panorama.fill_background import FillBackGround
from panorama.matcher import matcher
from panorama.StitchPanorama import StitchPanorama
from panorama.video import Video

panoramas = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("-fg", "--fgmode", default=Video.FG_MOG2)
    parser.add_argument("-bs", "--mv_blocksize", default=16)
    parser.add_argument("-k", "--mv_k", default=16)
    parser.add_argument("-th", "--mv_threshold", default=15)
    parser.add_argument("-c",
                        "--clear",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    return parser.parse_args()


def main(config: argparse.Namespace) -> None:
    with Video(config.filepath) as cap:
        if config.clear:
            print('Clearing file cache...')
            for f in glob.glob(f"{cap.filename}_*"):
                os.remove(f)

        fg, bg, fgmasks = cap.extract_foreground(config.fgmode, config)
        cap.write(f'{cap.filename}_fg', fg, cap.width, cap.height)

        panoFile = f'{cap.filename}_pano.jpg'
        if not os.path.exists(panoFile):
            # remove foreground and fill out the removed part in background
            # this issue involved camera motion, size change, object tracking
            fbg = FillBackGround()
            sampleBG = fbg.fill_background(bg, fgmasks, cap.fps)
            # using processed background and stitch them together to create panorama
            # we just need to sample 5 points for stitching Q1 - Q5
            sp = StitchPanorama(sampleBG)
            cv2.imwrite(panoFile, sp.simpleStitch())
        else:
            print('Cached panorama file is used.')

        bg = cv2.imread(panoFile)
        frames = cap.mergeForeground(bg, fg, fgmasks)
        cap.write(f'{cap.filename}_result', frames, len(bg[0]), len(bg))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)

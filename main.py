import argparse
import os
import glob

import cv2
import numpy as np
from tqdm import tqdm

from panorama.fill_background import FillBackGround
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


def get_fg_cache(filename: str) -> np.ndarray:
    fg_cap = cv2.VideoCapture(f'{filename}_fg.mp4')
    fg = []
    while (fg_cap.isOpened()):
        ret, frame = fg_cap.read()
        if ret is True:
            fg.append(frame)
        else:
            break
    fg = np.array(fg)
    return fg


def main(config: argparse.Namespace) -> None:
    with Video(config.filepath) as cap:
        if config.clear:
            print('Clearing file cache...')
            for f in glob.glob(f"{cap.filename}_*"):
                os.remove(f)

        fg, bg, fgmasks = cap.extract_foreground(config.fgmode, config)
        cap.write(f'{cap.filename}_fg', fg, cap.width, cap.height)
        # fg = get_fg_cache(cap.filename)

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
        out2, out1 = cap.mergeForeground(bg, fg)
        cv2.imwrite(f'{cap.filename}_out1.jpg', out1)
        cap.write(f'{cap.filename}_result', out2, bg.shape[1], bg.shape[0])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)

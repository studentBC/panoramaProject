import argparse
import os
import glob

import cv2
import numpy as np
from tqdm import tqdm

from panorama.fill_background import FillBackGround
from panorama.StitchPanorama import StitchPanorama
from panorama.video import Video
from panorama.draw_line import DrawLineWidget

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


def get_video_cache(filename: str) -> np.ndarray:
    cap = cv2.VideoCapture(filename)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            frames.append(frame)
        else:
            break
    frames = np.array(frames)
    return frames


def main(config: argparse.Namespace) -> None:
    with Video(config.filepath) as cap:
        if config.clear:
            print('Clearing file cache...')
            for f in glob.glob(f"{cap.filename}_*"):
                os.remove(f)

        fg, bg, fgmasks = cap.extract_foreground(config.fgmode, config)
        cap.write(f'{cap.filename}_fg', fg, cap.width, cap.height)
        # fg = get_video_cache(f'{cap.filename}_fg.mp4')

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
        res, out1 = cap.mergeForeground(bg, fg)
        cv2.imwrite(f'{cap.filename}_out1.jpg', out1)
        cap.write(f'{cap.filename}_result', res, bg.shape[1], bg.shape[0])

        # res = get_video_cache(f'{cap.filename}_result.mp4')
        print(
            'Draw a line to indicate the direction of camera motion and press q to leave...'
        )
        camera = DrawLineWidget(bg, res)
        while True:
            cv2.imshow(camera.window_name, camera.show_image())
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyWindow(camera.window_name)
                break
        out2 = cap.createNewCamera(bg, res, camera.image_coordinates[0],
                                   camera.image_coordinates[1])
        cap.write(f'{cap.filename}_out2', out2, cap.width, cap.height)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)

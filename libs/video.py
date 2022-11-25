import cv2
import numpy as np
from libs.foreground_extraction import ForegroundExtractor


class Video:
    FG_GRABCUT = "grabcut"
    FG_MOG = "mog"
    FG_MOG2 = "mog2"
    FG_GSOC = "gsoc"
    FG_GMG = "gmg"
    FG_HOG = "hog"
    FG_DOF = "dof"
    FG_LKO = "lko"
    FG_MV = "mv"  # motion vector
    FG_DST = "dst"

    def __init__(self, filepath: str) -> None:
        self._cap = cv2.VideoCapture(filepath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._cap.release()

    def set_backgound(self, background: any) -> None:
        return

    def write(self, filename: str) -> None:
        pass
        # display your foreground objects as a video sequence against a white plain background frame by frame.
        # https://www.etutorialspoint.com/index.php/319-python-opencv-overlaying-or-blending-two-images
        # for i in range(frame_count):
        #     #print(len(foreGround[i]), len(foreGround[i][0]))
        #     #print(len(panoramas[i]), len(panoramas[i][0]))
        #     new_h, new_w, channels = panoramas[i].shape
        #     resize = cv2.resize(fg[i], (new_w, new_h))
        #     dst = cv2.addWeighted(resize, 0.5, panoramas[i], 0.7, 0)
        #     fianlFrame.append(dst)

        # # Create a video a new video by defining a path in the panorama image, the foreground objects move in time synchronized manner.
        # # save video
        # video=cv2.VideoWriter('result.mp4', -1,fps,(width,height))
        # #sv = cv2.VideoWriter('./tmp/result.mp4', -1, fps, (height, width))
        # for f in fianlFrame:
        #     video.write(f)
    @property
    def fps(self) -> int:
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps =  int(cap.get(cv2.CAP_PROP_FPS))
        return int(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def frames(self) -> np.ndarray:
        frames = []
        while (self._cap.isOpened()):
            ret, frame = self._cap.read()
            if ret is True:
                frames.append(frame)
            else:
                break

        return np.array(frames)

    def extract_foreground(
            self, mode: str,
            config: any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        print("Extracting foreground...")
        fgmasks = []
        extractor = ForegroundExtractor()
        frames = self.frames

        if mode == Video.FG_GRABCUT:
            fgmasks = extractor.get_foreground_mask_grabcut(frames)
        elif mode == Video.FG_MOG:
            fgmasks = extractor.get_foreground_mask_mog(frames)
        elif mode == Video.FG_MOG2:
            fgmasks = extractor.get_foreground_mask_mog2(frames)
        elif mode == Video.FG_GSOC:
            fgmasks = extractor.get_foreground_mask_gsoc(frames)
        elif mode == Video.FG_GMG:
            fgmasks = extractor.get_foreground_mask_gmg(frames)
        elif mode == Video.FG_HOG:
            fgmasks = extractor.get_foreground_mask_hog(frames)
        elif mode == Video.FG_DOF:
            fgmasks = extractor.get_foreground_mask_dof(frames)
        elif mode == Video.FG_MV:
            fgmasks = extractor.get_foreground_mask_mv(
                frames, int(config.mv_blocksize), int(config.mv_k),
                float(config.mv_threshold))
        elif mode == Video.FG_DST:
            fgmasks = extractor.get_foreground_mask_dst(frames)
        else:
            raise Exception("Invalid fgmode")

        bgmasks = np.where((fgmasks == 1), 0, 1).astype('uint8')
        print(frames.shape, fgmasks.shape, fgmasks[:, :, :, np.newaxis].shape)

        fg = frames * fgmasks[:, :, :, np.newaxis]
        bg = frames * bgmasks[:, :, :, np.newaxis]

        return fg, bg, fgmasks

    def show(self, frames: np.ndarray) -> None:
        for frame in frames:
            cv2.imshow('frame', frame)
            # & 0xFF is required for a 64-bit system
            if cv2.waitKey(1000 // self.fps) & 0xFF == ord('q'):
                break

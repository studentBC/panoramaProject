import cv2
import numpy as np
from tqdm import tqdm

from panorama.foreground_extraction import ForegroundExtractor


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
        self._background = np.zeros(shape=[self.width, self.height, 3],
                                    dtype=np.uint8)
        self.filename = filepath.split('/')[-1].split('.')[0]
        self._frames = np.array([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._cap.release()

    def set_background(self, background: np.ndarray) -> None:
        self._background = background

    def mergeForeground(self, bg: np.ndarray, fg: np.ndarray,
                        fgmask: np.ndarray) -> list[np.ndarray]:
        print('merge panorama and foreground...')
        frames = []
        for i in tqdm(range(len(fg))):
            res = cv2.matchTemplate(bg, self.frames[i], cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            frame = self.overlay_image_alpha(bg, fg[i], max_loc[0], max_loc[1],
                                             fgmask[i])
            frames.append(frame)
        return frames

    def write(self, filename: str, frames: list[np.ndarray] | np.ndarray,
              w: int, h: int) -> None:
        file = cv2.VideoWriter(f'{filename}.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                               (w, h))
        for frame in frames:
            file.write(frame)
        file.release()

    @property
    def fps(self) -> int:
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return int(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frames(self) -> np.ndarray:
        if len(self._frames) > 0:
            return self._frames

        frames = []
        while (self._cap.isOpened()):
            ret, frame = self._cap.read()
            if ret is True:
                frames.append(frame)
            else:
                break
        self._frames = np.array(frames)

        return self._frames

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
        # print(frames.shape, fgmasks.shape, fgmasks[:, :, :, np.newaxis].shape)

        fg = frames * fgmasks[:, :, :, np.newaxis]
        bg = frames * bgmasks[:, :, :, np.newaxis]

        return fg, bg, fgmasks

    def show(self, frames: np.ndarray) -> None:

        for frame in frames:
            cv2.imshow('frame', frame)
            # & 0xFF is required for a 64-bit system
            if cv2.waitKey(1000 // self.fps) & 0xFF == ord('q'):
                break

    def overlay_image_alpha(self, img: np.ndarray, overlay: np.ndarray, x: int,
                            y: int, alpha_mask: np.ndarray) -> np.ndarray:
        # Image ranges
        img = img.copy()
        y1, y2 = max(0, y), min(img.shape[0], y + overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + overlay.shape[1])

        # Overlay ranges
        # y1o, y2o = max(0, -y), min(overlay.shape[0], img.shape[0] - y)
        # x1o, x2o = max(0, -x), min(overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2:
            return img

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        mask = alpha_mask[:, :, np.newaxis]
        # img_overlay_crop = overlay[y1o:y2o, x1o:x2o]
        mask_inv = 1.0 - mask

        img_crop[:] = mask * overlay + mask_inv * img_crop
        return img
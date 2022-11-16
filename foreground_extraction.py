import cv2
import numpy as np

from motionVectorCalculator import motionVector
from tqdm import tqdm
from imutils.object_detection import non_max_suppression

class ForegroundExtractor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def get_foreground_mask_grabcut(self, frames):
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = np.zeros(frame.shape[:2], np.uint8)
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            rect = (0, 0, frame.shape[1], frame.shape[0]) # width, height
            cv2.grabCut(frame, fgmask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

            fgmask = np.where((fgmask == 2) | (fgmask == 0), 0, 1).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_mog(self, frames):
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_mog2(self, frames):
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_gsoc(self, frames):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        fgmasks = []
        i = 0
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_gmg(self, frames):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)
    #get motion vector
    def get_foreground_mask_mv(self, frames, bs=16, k=16, threshold=10):
        frame_count, height, width, _ = frames.shape
        frames_yuv = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) for frame in frames])
        fgmasks = [np.zeros((height, width), np.uint8)]

        mv = motionVector()

        for fn in tqdm(range(1, frame_count)):
            fgmask = np.zeros((height, width), np.uint8)
            for y in range(0, height, bs):
                for x in range(0, width, bs):
                    bw = bs if x + bs <= width else width - x
                    bh = bs if y + bs <= height else height - y
                    dir_y, dir_x = mv.getMAD(frames_yuv[fn-1], frames_yuv[fn], y, x, bh, bw, k)
                    if dir_y ** 2 + dir_x ** 2 > threshold ** 2:
                        fgmask[y: y+bh, x: x+bw] = 1
            fgmasks.append(fgmask)

        return np.array(fgmasks)

    #https://pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    #https://thedatafrog.com/en/articles/human-detection-video/
    def get_foreground_mask_hog(self, frames):
        # detect people in the image
        fgmasks = []
        for frame in tqdm(frames):
            #rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8) )
            (rects, weights) = self.hog.detectMultiScale(frame, winStride=(8, 8),
		                                            padding=(2, 2), scale=1.05)
            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.imshow("lol", frame)
            # cv2.waitKey(500)
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            fgmask = np.zeros(frame.shape[:2], np.uint8)

            for (xA, yA, xB, yB) in pick:
                # draw the final bounding boxes
                # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                # apply GrabCut using the the bounding box segmentation method
                # try to cut human off from each box
                bgdModel = np.zeros((1,65), np.float64)
                fgdModel = np.zeros((1,65), np.float64)
                rect = (xA, yA, xB-xA, yB-yA)
                cv2.grabCut(frame, fgmask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            fgmask = np.where((fgmask == 2) | (fgmask == 0), 0, 1).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

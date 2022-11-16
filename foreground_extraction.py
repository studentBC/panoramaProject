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
    def get_foreground_mask_mv(self, frames, k):
        #print("enter get_foreground_mask_mv")
        fgmasks = []
        end = len(frames)
        mv = motionVector()
        #we have 16*16 block or k*k
        for a in tqdm(range(1, end)):
            block = [] # list of pair of value, motion vector
            for i in range(0, frames[a].shape[0], 16):
                for j in range(0, frames[a].shape[1], 16):
                    #calculate one block value 
                    # YCrCb = cv2.cvtColor(frames[a], cv2.COLOR_BGR2YCrCb)
                    # Y, Cr, Cb = cv2.split(YCrCb)
                    value, vector = mv.getMAD(i, j, cv2.cvtColor(frames[a], cv2.COLOR_BGR2YCrCb)[:,:,0] 
                                                    , cv2.cvtColor(frames[a-1], cv2.COLOR_BGR2YCrCb)[:,:,0], k)
                    block.append([value, vector, (i, j)])
            #sort by value to determine which motion vector belongs to background
            #we determine by the largest different between two sequence pls note that this soluiton can only apply
            #for single object moving in a static background
            block.sort()
            diff = block[1][0]-block[0][0]
            threshold = (block[-1][0]-block[0][0])/2
            fgmask = np.zeros(frames[a].shape[:2], np.uint8)
            print("we get threshold: ", threshold)
            #the difference range should not exceed a threshold
            for i in range(2, end):
                if block[i][0] - block[i-1][0] > threshold:
                    #find the foreground start index
                    index = i
                    for j in range(i, end):
                        #start to mask the macroblock 
                        fgmask[block[j][2][0]: block[j][2][0]+k, block[j][2][1]:block[j][2][1]+k , 0] = 1
                    fgmasks.append(fgmask)
                    break
            #since we will lost one frame in the very beginning or end so we just insert one frame in the beginning
        fgmasks.insert(0, fgmasks[0])
        fgmasks.append(fgmask)

        return fgmasks
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

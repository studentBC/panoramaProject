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
        #motion_vectors = [] to record background motion vector
        end = len(frames)
        mv = motionVector()
        #we have 16*16 block or k*k
        for a in tqdm(range(1, end)):
            block = [] # list of pair of value, motion vector
            prevFrame = cv2.cvtColor(frames[a-1], cv2.COLOR_BGR2YCrCb)[:,:,0]
            curFrame = cv2.cvtColor(frames[a], cv2.COLOR_BGR2YCrCb)[:,:,0]
            for i in range(0, frames[a].shape[0], k):
                for j in range(0, frames[a].shape[1], k):
                    #calculate one block value 
                    # YCrCb = cv2.cvtColor(frames[a], cv2.COLOR_BGR2YCrCb)
                    # Y, Cr, Cb = cv2.split(YCrCb)
                    value, vector = mv.getMAD(i, j,  curFrame, prevFrame, k)
                    block.append([value, vector, (i, j)])
            #sort by value to determine which motion vector belongs to background
            #we determine by the largest different between two sequence pls note that this soluiton can only apply
            #for single object moving in a static background
            block.sort()
            diff = block[1][0]-block[0][0]
            threshold = (block[-1][0]-block[0][0])/2
            fgmask = np.zeros(frames[a].shape[:2], np.uint8)
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
    def get_foreground_mask_dof(self, frames):
        prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frames[0])
        hsv[..., 1] = 255
        fgmasks = []
        #print("enter get_foreground_mask_dof")
        for i in tqdm(range(1, len(frames))):
            next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #h, s, v1 = cv2.split(bgr)
            #cv2.imshow("gray-image",bgr[:,:, 2:3])
            #now we convert every fram to 0 or 255
            (thresh, im_bw) = cv2.threshold(bgr[:,:, 2:3], 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imshow('frame2', im_bw)
            # k = cv2.waitKey(30) & 0xff
            # replay any > 0 to 1
            im_bw[im_bw > 0] = 1
            prvs = next
            fgmasks.append(im_bw)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        
        # for frame in tqdm(frames):
        #     fgmask = fgbg.apply(frame)
        #     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        #     fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
        #     fgmasks.append(fgmask)
        fgmasks.append(fgmasks[-1])
        return np.array(fgmasks)
        
    def get_foreground_mask_lko(self, frames):
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        old_frame = frames[0]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        for i in tqdm(range(1, len(frames))):

            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frames[i] = cv2.circle(frames[i], (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frames[i], mask)
            cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

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

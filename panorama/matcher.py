import cv2
import numpy as np
from matplotlib import pyplot as plt


class matcher:

    def __init__(self):
        #self.surf = cv2.SURF_create()
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, i1, i2, direction=None):
        imageSet1 = self.getSIFTFeatures(i1)
        imageSet2 = self.getSIFTFeatures(i2)
        matches = self.flann.knnMatch(imageSet2['des'], imageSet1['des'], k=2)
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append((m.trainIdx, m.queryIdx))

        if len(good) > 4:
            pointsCurrent = imageSet2['kp']
            pointsPrevious = imageSet1['kp']

            matchedPointsCurrent = np.float32(
                [pointsCurrent[i].pt for (__, i) in good])
            matchedPointsPrev = np.float32(
                [pointsPrevious[i].pt for (i, __) in good])

            H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev,
                                      cv2.RANSAC, 4)
            return H
        return None

    def getSIFTFeatures(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return {'kp': kp, 'des': des}

    def searchTransformation(self, src: np.ndarray,
                             dst: np.ndarray) -> np.ndarray | None:
        srcFeatures = self.getSIFTFeatures(src)
        dstFeatures = self.getSIFTFeatures(dst)
        matches = self.flann.knnMatch(srcFeatures['des'],
                                      dstFeatures['des'],
                                      k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > 4:
            src_pts = np.float32([
                srcFeatures['kp'][m.queryIdx].pt for m in good
            ]).reshape(-1, 1, 2)
            dst_pts = np.float32([
                dstFeatures['kp'][m.trainIdx].pt for m in good
            ]).reshape(-1, 1, 2)
            transformation, _ = cv2.findHomography(dst_pts, src_pts,
                                                   cv2.RANSAC, 5.0)
            return transformation
        return None
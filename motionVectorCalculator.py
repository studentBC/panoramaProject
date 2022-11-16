import cv2
import sys
import math
import numpy as np

class motionVector:
    def getMAD(self, x, y, cur, prev, k):
        #our search distance is 2K+1 for k is given by input
        #our right point will be i+16, j+16
        w, h = cur.shape[1], cur.shape[0]
        ex, ey = min(x+k, w), min(y+k, h)
        #searching block area
        #left upper point and right down point of searching block area
        xstart, ystart, xend, yend = max(0, x-k), max(y-k, 0), min(ex+k, w), min(ey+k, h) 
        #print(x, y, xstart, ystart, xend, yend)
        total, small, avg = 16*16, sys.maxsize, 0
        targetVector = []
        for a in range(ystart, yend):
            for b in range(xstart, xend):
                if a == x and y == b:
                    continue
                c, d, sum = a, b, 0
                for i in range(y, ey):
                    for j in range(x, ex):
                        sum+=abs(cur[i][j]-prev[c][d])/total #prevent overflow problem
                        d+=1
                    c+=1
                #avg = sum/total
                avg = sum
                #print(sum, avg)
                if small > avg:
                    small = avg
                    targetVector = [a, b]
        return small, targetVector
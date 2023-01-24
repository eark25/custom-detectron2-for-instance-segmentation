from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    # print('p: ', p)
    q = list(q_)
    # print('q: ', q)
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians ** its give pi for atan2 (p <----- q)  measure from x clockwise
    # print('atan2: ', angle)
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # print(hypotenuse)

    # Here we lengthen the arrow by a factor of scale (p ----- q * scale)
    # every vector will be reversed because the angle calculated from point q to p when it should be p to q
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    # print('new q: ', q)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # create the arrow hooks from point q (p -----> q)
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    
def getOrientation(index, pts, img):
    # count the point
    sz = len(pts)
    # print(sz)
    # create array using sz
    data_pts = np.empty((sz, 2), dtype=np.float64)
    # print(data_pts.shape)
    for i in range(data_pts.shape[0]): # range(sz)
        # x
        data_pts[i,0] = pts[i,0,1]
        # y
        data_pts[i,1] = pts[i,0,0]
    # print(data_pts)
    # Perform PCA analysis
    # print(data_pts)
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    # print(mean)
    # print(eigenvectors)
    # print(eigenvectors[0,0])
    # print(eigenvectors[0,1])
    # print(eigenvectors[1,0])
    # print(eigenvectors[1,1])
    # print(eigenvalues)
    # Store the center of the object (x, y)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    # print(cntr)
    # draw a circle at the center
    cv.circle(img, cntr, 3, (255, 0, 255), 2) # purple
    cv.putText(img, str(index), (cntr[0] - int(20), cntr[1] + int(20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv.LINE_AA)
    # principle component 1 (x, y)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    # cv.circle(img, (int(p1[0]), int(p1[1])), 3, (255, 0, 255), 2)
    # print('q of pca1: ', p1)
    # principle component 2
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    # cv.circle(img, (int(p2[0]), int(p2[1])), 3, (255, 0, 255), 2)
    # print('q of pca2: ', p2)
    drawAxis(img, cntr, p1, (0, 255, 0), 1) # green
    drawAxis(img, cntr, p2, (255, 255, 0), 5) # cyan
    # drawAxis(img, cntr, (cntr[0] + 100, cntr[1]), (0, 255, 255),100) # yellow
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    # print(angle)
    
    return angle

def getOutputOrientation(masks, img):
    # Check if image is loaded successfully
    if masks is None:
        print('Could not open or find any instance: ')
        exit(0)

    angles = []
    for i in range(masks.shape[0]):
        mask = masks[i].to("cpu").long()
        # print('mask: ', mask)
        # print(np.unique(mask))
        # print(mask.shape)
        # exit(0)
        # Convert image to grayscale
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Convert image to binary
        # _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        obj = np.expand_dims(np.nonzero(mask), axis=1) # not using np.transpose because we are not using cv.imread()
        # print('obj: ', obj)
        # print(obj.shape)
        try:
            assert len(obj) != 0, 'no crack detected'
        except AssertionError as e:
            # import sys
            # sys.exit(0)
            continue
        # print(obj) # swap this
        # print(obj.shape)
        angle = getOrientation(i+1, obj, img)
        angles.append(-(angle * 180 / pi))
        # print('angle of crack {}:'.format(i+1), -(angle * 180 / pi), 'degree')
        # maybe use contour from prediction
        # contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        # print(contours)
        # print(contours[0].shape)
        # print(contours[0][1,0,0])
        # print(src.shape) # opencv shape (H, W, C)
        # print('here:', obj[0].shape)
        # print(obj[0][0][1])
        # for i in obj:
        #     cv.circle(src, (i[0][1], i[0][0]), 1, (255, 0, 255), 2)
        # for i, c in enumerate(contours):
        #     # Calculate the area of each contour
        #     area = cv.contourArea(c)
        #     # Ignore contours that are too small or too large
        #     if area < 1e2 or 1e5 < area:
        #         continue
        #     # Draw each contour only for visualisation purposes
        #     # cv.drawContours(src, contours, i, (0, 0, 255), 2)
        #     # Find the orientation of each shape
        #     getOrientation(i + 1, c, src)
        #     # break
    return img, angles
    # cv.imwrite('pca_crack.jpg', src)

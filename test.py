import cv2 as cv
import numpy as np

img = cv.imread('data/f02/image/0028.png', 0)
img = cv.resize(img, (240, 560))

img2 = np.power(img/float(np.max(img)), 3)
img2[:, 0:60] = 0
img2[:, 180:240] = 0

img3 = np.power(img/float(np.max(img)), 3.5)
img3[:, 0:60] = 0
img3[:, 180:240] = 0
#print(img2)

cv.imshow('ori', img)
cv.imshow('gamma', img2)
cv.imshow('gamma2', img3)
cv.waitKey(0)
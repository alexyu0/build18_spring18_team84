import cv2
import numpy as np
import imutils
 
img = cv2.imread('object_2.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# [0, 26, 0] [174, 255, 49]
lower_range = np.array([108, 100, 100], dtype=np.uint8)
upper_range = np.array([128, 255, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower_range, upper_range)

cv2.imshow('image', img)
cv2.imshow('mask', mask)

while True:
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break
 
cv2.destroyAllWindows()

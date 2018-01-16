import cv2
import numpy as np
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray.resize()

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


img = cv2.imread('picture.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows,cols,ch = img.shape

#corner detection part:
dst = cv2.cornerHarris(gray, 2,3,0.04)
dst = cv2.dilate(dst, None)
print(dst.shape)
# img[dst > 0.01*dst.max()] = [255,0,0]
dst[dst <= 0.01*dst.max()] = 0
cornerPoints = np.nonzero(dst)
cornerPoints = zip(*cornerPoints[::-1])

topLeft = [((x-0)**2 + (y-0)**2) for (y,x) in cornerPoints]
topLeft = cornerPoints[topLeft.index(min(topLeft))]
topRight = [((x-cols)**2 + (y-0)**2) for (y,x) in cornerPoints]
topRight = cornerPoints[topRight.index(min(topRight))]
botLeft = [((x-0)**2 + (y-rows)**2) for (y,x) in cornerPoints]
botLeft = cornerPoints[botLeft.index(min(botLeft))]
botRight = [((x-cols)**2 + (y-rows)**2) for (y,x) in cornerPoints]
botRight = cornerPoints[botRight.index(min(botRight))]

print(topLeft)
img[topLeft[0],topLeft[1]] = [255,0,0]
cv2.circle(img, (topLeft[0], topLeft[1]), 5, [255,0,0])
cv2.circle(img, (topRight[0], topRight[1]), 5, [255,0,0])
cv2.circle(img, (botLeft[0], botLeft[1]), 5, [255,0,0])
cv2.circle(img, (botRight[0], botRight[1]), 5, [255,0,0])

# pts1 = np.float32([[26,10],[245,35],[26,240],[245,221]])
# pts1 = np.float32([[28,10],[215,56],[26,233],[216,166]])
pts1 = np.float32([topLeft[::], botLeft[::], topRight[::], botRight[::]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
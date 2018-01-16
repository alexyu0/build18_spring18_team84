import cv2
import numpy as np 
import matplotlib.pyplot as plt

#filename: path to the desired file
#returns original image, grayscale image, height, width
def readImage(filename):
	img = cv2.imread(filename)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rows, cols, ch = img.shape
	return img, gray, rows, cols
#img: grayscale image
#returns an array of corner locations
def cornerDetect(gray):
	dst = cv2.cornerHarris(gray, 2,3,0.04)
	dst = cv2.dilate(dst, None)
	# img[dst > 0.01*dst.max()] = [255,0,0]
	dst[dst <= 0.01*dst.max()] = 0
	cornerPoints = np.nonzero(dst)
	cornerPoints = zip(cornerPoints[1],cornerPoints[0])
	return cornerPoints

def getClosestTo(cornerPoints, y_0, x_0):
	loc = [((x-x_0)**2 + (y-y_0)**2) for (y,x) in cornerPoints]
	loc = cornerPoints[loc.index(min(loc))]
	return loc

#cornerPoints: array of corner locations (row, col)
#returns the four points closest to the topleft, topright, botleft and botright
def getCorners(cornerPoints, rows, cols):
	topLeft = getClosestTo(cornerPoints, 0, 0)
	topRight = getClosestTo(cornerPoints, 0, cols)
	botLeft = getClosestTo(cornerPoints, rows, 0)
	botRight = getClosestTo(cornerPoints, rows, cols)
	return topLeft, topRight, botLeft, botRight

def perspectiveTrans(img,  width, height, topLeft, topRight, botLeft, botRight):
	pts1 = np.float32([topLeft, botLeft, topRight, botRight])
	pts2 = np.float32([[0,0],[height,0],[0,width],[height,width]])
	M = cv2.getPerspectiveTransform(pts1,pts2)

	dst = cv2.warpPerspective(img,M,(height,width))
	return dst

def show(img1, img2):
	plt.subplot(121),plt.imshow(img1),plt.title('Input')
	plt.subplot(122),plt.imshow(img2),plt.title('Output')
	plt.show()




def stillImage():
	img, gray, rows, cols = readImage("picture.jpeg")
	cornerPoints = cornerDetect(gray)
	topLeft, topRight, botLeft, botRight = getCorners(cornerPoints, rows, cols)
	res = (480,720)
	transformedImg = perspectiveTrans(img, res[0],res[1], topLeft, topRight, botLeft, botRight)

	thickness = 2
	radius = 5
	color = [255,0,0]
	cv2.circle(img, topLeft, radius, color, thickness)
	cv2.circle(img, topRight, radius, color, thickness)
	cv2.circle(img, botLeft, radius, color, thickness)
	cv2.circle(img, botRight, radius, color,thickness)


	show(img, transformedImg)

def webcam():
	cap = cv2.VideoCapture(0)
	while (True):
		ret, frame = cap.read()
		ratio = 0.3
		frame = cv2.resize(frame, (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio)))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		cornerPoints = cornerDetect(gray)
		topLeft, topRight, botLeft, botRight = getCorners(cornerPoints, gray.shape[1], gray.shape[0])
		res = (frame.shape[0], frame.shape[1])
		transformedImg = perspectiveTrans(gray, res[0],res[1], topLeft, topRight, botLeft, botRight)

		thickness = 2
		radius = 5
		color = [255,0,0]
		for point in cornerPoints:
			cv2.circle(frame, (point[0], point[1]), radius, [0,0,255], thickness)

		cv2.circle(frame, topLeft, radius, color, thickness)
		cv2.circle(frame, topRight, radius, color, thickness)
		cv2.circle(frame, botLeft, radius, color, thickness)
		cv2.circle(frame, botRight, radius, color,thickness)


		cv2.imshow('frame', frame)
		cv2.imshow('frame2', transformedImg)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
	
def main():
	webcam()

if __name__ == "__main__":
	main()
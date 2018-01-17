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
	# print(dst[0])
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

def webcam_line():
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	ratio = 0.3
	print(frame.shape)
	width, height = (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio))
	print(height, width)
	img = np.zeros((height*3,width ,3), np.uint8)

	mode = "auto"
	#mouse bindings
	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame',get_click)

	while (True):
		ret, frame = cap.read()
		ratio = 0.3
		frame = cv2.resize(frame, (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio)))
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		if mode == "auto":
			edges = cv2.Canny(gray,50,100,apertureSize = 3)
			# lines = cv2.HoughLines(edges,1,np.pi/180,100)
			minLineLength = 100
			maxLineGap = 10
			lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
			if lines is not None:
				for line in lines:
					for x1,y1,x2,y2 in line:
					    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
			# if lines is not  None:
			# 	for i in range(len(lines)):
			# 		for rho,theta in lines[i]:
			# 		    a = np.cos(theta)
			# 		    b = np.sin(theta)
			# 		    x0 = a*rho
			# 		    y0 = b*rho
			# 		    x1 = int(x0 + 1000*(-b))
			# 		    y1 = int(y0 + 1000*(a))
			# 		    x2 = int(x0 - 1000*(-b))
			# 		    y2 = int(y0 - 1000*(a))

			#     		cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)


			cornerPoints = cornerDetect(edges)
			topLeft, topRight, botLeft, botRight = getCorners(cornerPoints, edges.shape[1], edges.shape[0])
		

		res = (edges.shape[0], edges.shape[1])
		transformedImg = perspectiveTrans(frame, res[0],res[1], topLeft, topRight, botLeft, botRight)


		thickness = 2
		radius = 5
		color = [255,0,0]
		# for point in cornerPoints:
		# 	cv2.circle(frame, (point[0], point[1]), radius, [0,0,255], thickness)
		cv2.circle(frame, topLeft, radius, color, thickness)
		cv2.circle(frame, topRight, radius, color, thickness)
		cv2.circle(frame, botLeft, radius, color, thickness)
		cv2.circle(frame, botRight, radius, color,thickness)

		

		# im
		# cv2.imshow('frame', frame)
		# cv2.imshow('canny', edges)
		# cv2.imshow('fram2', transformedImg)
		img[:height][:] = frame
		img[height:2*height][:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
		img[2*height:][:] = transformedImg
		cv2.imshow('frame',img)
		k = cv2.waitKey(30)
		if  k & 0xFF == ord('q'):
			break
		if k == ord('m'):
			mode = "manual"
   			x = manual_calibrate(cap, ratio)
   			print(x)
   			if x is -2:
   				break
   			elif x is -1:
   				continue
   			else:
   				# topLeft, topRight, botLeft, botRight = x
   				topLeft = (int(x[0][0]*ratio),int(x[0][1]*ratio))
   				topRight = (int(x[1][0]*ratio),int(x[1][1]*ratio))
   				botLeft = (int(x[2][0]*ratio),int(x[2][1]*ratio))
   				botRight = (int(x[3][0]*ratio),int(x[3][1]*ratio))
   				print(topLeft, topRight, botLeft, botRight)
	cap.release()
	cv2.destroyAllWindows()

mx,my = -1,-1
mouse_time = -1
import time
def get_click(event, x, y, flags, param):
	global mx, my, mouse_time
	if event == cv2.EVENT_LBUTTONDOWN:
		mx, my = x, y
		mouse_time = time.time()

def manual_calibrate(cap, ratio):
	global mx, my, mouse_time

	corners = ["topLeft", "topRight", "botLeft", "botRight"]
	cornerPoints = [None, None, None, None]
	i = 0
	cmx, cmy = mx, my 
	mt = mouse_time
	thickness = 2
	radius = 5
	color = [255,0,0]

	while True:
		ret, frame = cap.read()
		# frame = cv2.resize(frame, (int(frame.shape[1]*ratio), int(frame.shape[0]*ratio)))
		
		#draw the corners
		for point in range(i):
			cv2.circle(frame, cornerPoints[point], radius, color, thickness)

		#check if the mouse click position has changed
		if mouse_time != mt:
			print("asdfasdf")
			cornerPoints[i] = (mx, my)
			mt = mouse_time
			i += 1
			if i == 4:
				# return list(map(lambda x:[x[1],x[0]], cornerPoints))
				return cornerPoints

		cv2.putText(frame, "select %s" % corners[i], (0,frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0])
		cv2.imshow('frame', frame)

		k = cv2.waitKey(30)
		if k & 0xFF == ord('q'):
			return -2
		if k == ord('z'):
   			return -1
def main():
	webcam_line()

if __name__ == "__main__":
	main()



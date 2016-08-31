import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def shiftLocation(shape, location):
	mx = np.mean(shape[::2])
	my = np.mean(shape[1::2])
	shape[::2] = shape[::2]-mx+location[0]
	shape[1::2] = shape[1::2]-my+location[1]

def makeStrip(points, n, k):
	# strips of length 2k+1 at point n(indexed from 0); k index from 1
	X = points[::2]
	Y = points[1::2]
	p1 = (X[n], Y[n])
	p0 = (X[-1], Y[-1]) if(n==0) else (X[n-1], Y[n-1])
	p2 = (X[n-39], Y[n-39]) if((n+1)%40==0) else (X[n+1], Y[n+1])
	V = [-1.*float(p2[1] - p0[1]), float(p2[0] - p0[0])]
	U = np.divide(V, np.linalg.norm(V))
	
	X0 = (p1[0] + float(k)*U[0], p1[1] + float(k)*U[1])
	X1 = (p1[0] - float(k)*U[0], p1[1] - float(k)*U[1])

	line = []
	for i in range(k):	
		line.append((p1[0] + float(i)*U[0], p1[1] + float(i)*U[1]))
		line.append((p1[0] - float(i)*U[0], p1[1] - float(i)*U[1]))
	line = sorted(list(set(line)))

	"""	
	deltaErr = abs(float(X0[1]-X1[1])/float(X0[0]-X1[0]))
	error = -1.0
	line = []
	y = int(X0[1])
	(x0, x1) = (int(X0[0]), int(X1[0])) if X0[0]<X1[0] else (int(X1[0]), int(X0[0]))
	for x in range(x0, x1):
		line.append((x,y))
		error += deltaErr
		if error>=0.0:
			y+=1
			error -= 1.0
	"""	
	for i in range(len(line)):
		line[i] = (int(line[i][0]), int(line[i][1]))
	return line
	return X0, p1, X1, line

def getGrayStrip(img, points):
	strip = []
	for p in points:
		strip.append(np.mean(img[p[1], p[0]]))
	for i in range(len(strip)-1):
		strip[i] = strip[i] - strip[i+1]
	strip = strip[:-1]
	absSum = sum([abs(x) for x in strip])
	strip = np.divide(strip, absSum)
	return strip

def grayModel(images, landmarks, point=25, profileLength=40):
	#shiftLocation(shape, location)
	strips = []
	for i in range(len(landmarks)):
		line = makeStrip(landmarks[i], point, profileLength)
		grayStrip = getGrayStrip(images[i], line)
		strips.append(grayStrip)
	grayMean = np.mean(strips, axis=0)
	Sg = np.cov(np.array(strips).T)
	return grayMean, Sg

def grayModels(images, landmarks, profileLength=40):
	return [ (grayModel(images, landmarks, i, profileLength)) for i in range(len(landmarks[0][::2]))]

def grayFit(img, shape, grayMean, Sg, point=25, profileLength=80):
	line = makeStrip(shape, point, profileLength)
	grayStrip = getGrayStrip(img, line)
	Fgs = []
	for i in range(len(grayStrip)-len(grayMean)+1):
		segment = grayStrip[i:(i+len(grayMean))]
		fgs = np.dot(np.subtract(segment, grayMean).T, np.linalg.pinv(Sg))
		fgs = np.dot(fgs, np.subtract(segment, grayMean))
		Fgs.append(fgs)
	return Fgs, line


def grayFitShape(img, shape, grayModels, profileLength=80, metric='min'):
	newShape = [0]*len(shape)
	for point in range(len(shape[::2])):
		Fgs, line = grayFit(img, shape, grayModels[point][0], grayModels[point][1], point, profileLength)
		if metric == 'min':
			x = line[Fgs.index(min(Fgs)) + (len(grayModels[point][0])/2)][0]
			y = line[Fgs.index(min(Fgs)) + (len(grayModels[point][0])/2)][1]
		else:
			x = line[Fgs.index(max(Fgs)) + (len(grayModels[point][0])/2)][0]
			y = line[Fgs.index(max(Fgs)) + (len(grayModels[point][0])/2)][1]
		newShape[point*2]=x
		newShape[(point*2)+1]=y
		
	
	#print "length of shape & newShape: ", len(shape), len(newShape)
	#print "length of Fgs", len(Fgs)
	#print "length of line", len(line)
	#print "length of gray model: ", len(grayModels[0][0])
	#print "derived length of Fgs: ", (2*profileLength+1) - len(grayModels[0][0]) + 1
	return newShape


def grayFitAll(images, landmarks):
	maxs = []; mins = []
	for point in range(len(landmarks[0][::2])):
	#for point in range(15,25) + range(55,65) + range(95,105) + range(135, 145):
		grayMean, Sg = grayModel(images, landmarks, point)
		for i in range(len(images)):
			Fgs, _ = grayFit(images[i], landmarks[i], grayMean, Sg, point)
			plt.plot([Fgs.index(min(Fgs))], [min(Fgs)], 'go')
			plt.plot([Fgs.index(max(Fgs))], [max(Fgs)], 'ro')
			maxs.append(Fgs.index(max(Fgs)))
			mins.append(Fgs.index(min(Fgs)))

	#plt.plot(Fgs)
	#plt.plot([min(Fgs)]*len(Fgs))
	#plt.plot([max(Fgs)]*len(Fgs))
	plt.figure(2)
	plt.hist(maxs)
	plt.figure(3)
	plt.hist(mins)
	plt.show()



"""
	cv2.circle(img, p0, 10, (0,30, 230), 3)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image',img)
	cv2.waitKey(0)
"""	
	

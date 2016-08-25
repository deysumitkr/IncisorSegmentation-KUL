from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

def meanCentering(landmarks):
	nlandmarks = []
	for l in landmarks:
		nlandmarks.append(copy.copy(l))
		mx = np.mean(l[::2])
		my = np.mean(l[1::2])
		for i in range(0, len(l), 2):
			nlandmarks[-1][i] -= mx
		for k in range(1,len(l), 2):
			nlandmarks[-1][k] -= my
	return [mx, my, nlandmarks]

def plotLandmarks(landmarks):
	for l in landmarks:
		plt.plot(l[::2], l[1::2])
	plt.gca().invert_yaxis()
	plt.show()

def transform(T, points):
	ipMat = [points[::2], points[1::2]]
	opMat = np.dot(T, ipMat)
	flat = [None]*(len(points))
	flat[::2] = opMat[0]
	flat[1::2] = opMat[1]
	return flat

def scaleRotate(landmarks):
	landmarks[0] = np.divide(landmarks[0], np.linalg.norm(landmarks[0])) # scale first shape to unit vector
	for j in range(1,len(landmarks)):
		a = np.divide(np.dot(landmarks[j], landmarks[0]), np.linalg.norm(landmarks[j])**2)
		bn = 0.
		for i in range(0, len(landmarks[j]), 2):
			bn += float(landmarks[j][i]*landmarks[0][i+1] - landmarks[0][i]*landmarks[j][i+1])
		b = bn/(np.linalg.norm(landmarks[j])**2)
		scale = math.sqrt(a**2 + b**2)
		theta = math.atan(b/a)
		R = [[math.cos(theta), math.sin(theta)], [-1.0*math.sin(theta), math.cos(theta)]]
		T = np.multiply(R, scale)
		landmarks[j] = transform(T, landmarks[j])
	return landmarks

def alignShape(landmarks):
	[mx, my, cLandmarks] = meanCentering(landmarks)
	cLandmarks = scaleRotate(cLandmarks)
	while (abs(mx) + abs(my))>0.0001:
		[mx, my, cLandmarks] = meanCentering(cLandmarks)
		cLandmarks = scaleRotate(cLandmarks)
	return cLandmarks
	

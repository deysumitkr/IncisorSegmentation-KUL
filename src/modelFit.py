import math
import numpy as np
import alignShape
import matplotlib.pyplot as plt

def getTransformationMatrix(target, model):
	#target = np.divide(target, np.linalg.norm(target)) # scale first shape to unit vector
	a = np.divide(np.dot(model, target), np.linalg.norm(model)**2)
	bn = 0.
	for i in range(0, len(model), 2):
		bn += float(model[i]*target[i+1] - target[i]*model[i+1])
	b = bn/(np.linalg.norm(model)**2)
	
	scale = math.sqrt(a**2 + b**2)
	theta = math.atan(b/a)
	R = [[math.cos(theta), math.sin(theta)], [-1.0*math.sin(theta), math.cos(theta)]]
	T = np.multiply(R, scale)
	return target, T
	
def limitB(b, eig_vals):
	for i in range(len(b)):
		lamb = abs(math.sqrt(eig_vals[i]))
		if(abs(b[i]) > 3.0*lamb):
			b[i] = 3.0*lamb if(b[i]>0.0) else -3.0*lamb

def plots(data, labels=[]):
	for i in range(len(data)):
		if type(data[i]) is not list and type(data[i]) is not np.ndarray:
			raise ValueError("List of lists Expected")
		l = labels[i] if i < len(labels) else ''
		plt.plot(data[i][::2], data[i][1::2], label=l)
	plt.gca().invert_yaxis()
	plt.legend()
	plt.show()

def translate(points, tx, ty):
	points[::2] = points[::2]+tx
	points[1::2] = points[1::2]+ty

def fit(targetShape, meanShape, eig_vals, eig_vecs, comp=5):
	P = np.array(eig_vecs[:comp]).T
	b = np.array([0]*comp)
	[tx, ty, cTargetShape] = alignShape.meanCentering([targetShape])

	for _ in range(50):
		modelShape = meanShape + np.dot(P, b)
		target, T = getTransformationMatrix(cTargetShape[0], modelShape)
		newTarget = alignShape.transform(np.linalg.pinv(T), target)
		yNew = np.divide(newTarget, np.dot(newTarget, meanShape))
		b = np.dot(np.array(P).T, yNew - meanShape)
		limitB(b, eig_vals)

	modelShape = meanShape + np.dot(P, b)
	modelShape = alignShape.transform(T, modelShape) 
	translate(modelShape, tx,ty)
	#plots([meanShape, modelShape, cTargetShape[0], targetShape], labels=['meanShape', 'model fit','centered target', 'actual Target'])
	return modelShape






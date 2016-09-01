import copy
import math
import numpy as np
import alignShape
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

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
		if(abs(b[i]) > 9.0*lamb):
			b[i] = 3.0*lamb if(b[i]>0.0) else -3.0*lamb

def plots(data, labels=[], title='', UD=''):
	for i in range(len(data)):
		if type(data[i]) is not list and type(data[i]) is not np.ndarray:
			raise ValueError("List of lists Expected")
		l = labels[i] if i < len(labels) else ''
		plt.plot(data[i][::2], data[i][1::2], label=l)
	plt.gca().invert_yaxis()
	plt.title(title)
	plt.legend()
	plt.grid(True)
	if UD!='':
		plt.savefig('report/shapeFitting{0}.png'.format(UD))
	plt.show()
		

def translate(points, tx, ty):
	points[::2] = points[::2]+tx
	points[1::2] = points[1::2]+ty

def fit(targetShape, meanShape, eig_vals, eig_vecs, comp=5, plot=False):
	P = np.array(eig_vecs[:comp]).T
	b = np.array([0]*comp)
	[tx, ty, cTargetShape] = alignShape.meanCentering([targetShape])

	modelShape = meanShape + np.dot(P, b)
	target, T = getTransformationMatrix(cTargetShape[0], modelShape)
	modelShape = alignShape.transform(T, modelShape) 
	translate(modelShape, tx,ty)
	#plots([modelShape, targetShape], labels=['Model fit', 'Target Shape'], UD='', title='Mean Squared Error: {0}'.format(mse(modelShape, targetShape)))

	allShapes = []
	for _ in range(100):
		modelShape = meanShape + np.dot(P, b)
		target, T = getTransformationMatrix(cTargetShape[0], modelShape)
		newTarget = alignShape.transform(np.linalg.pinv(T), target)
		yNew = np.divide(newTarget, np.dot(newTarget, meanShape))
		b = np.dot(np.array(P).T, yNew - meanShape)
		limitB(b, eig_vals)
		saveShape = alignShape.transform(T, modelShape) 
		translate(saveShape, tx,ty)
		allShapes.append(mse(saveShape, targetShape))

	modelShape = meanShape + np.dot(P, b)
	modelShape = alignShape.transform(T, modelShape) 
	translate(modelShape, tx,ty)

	if plot:
		plots([modelShape, targetShape], labels=['Model fit', 'Target Shape'], UD='', title='Mean Squared Error: {0}'.format(mse(modelShape, targetShape)))
		plt.figure(4)
		plt.plot(allShapes)
		plt.xlabel('Iterations')
		plt.ylabel('Mean Squared Error')
		plt.title('Convergence of model to the target shape')
		plt.grid(True)
		plt.savefig('report/shapeFitConvergence{0}{1}.png'.format(len(modelShape)/80, 'L'))
		plt.show()
	return modelShape






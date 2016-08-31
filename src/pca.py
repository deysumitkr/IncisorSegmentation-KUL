import math
import numpy as np
import matplotlib.pyplot as plt

def eigValsVecs(landmarks):
	mu = np.mean(landmarks, axis=0)
	cov = np.cov(np.array(landmarks-mu).T)
	eig_val, eig_vec = np.linalg.eigh(cov)
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	eig_val = [v[0] for v in eig_pairs]
	eig_vec = [v[1] for v in eig_pairs]
	return eig_val, eig_vec

def plotEigenValues(vals, comp=10):
	plt.figure(1)
	plt.plot(vals[:10], range(1,comp+1))
	plt.grid(True)
	plt.title('Eigenvalues'.format(comp))
	plt.ylabel('Number of Components')
	plt.savefig('report/eigenValues.png')
	
	cum_val = [0]
	for v in vals:
		cum_val.append(cum_val[-1]+v)
	del cum_val[0]
	cum_val = np.divide(cum_val, cum_val[-1])
	
	plt.figure(2)
	plt.plot(cum_val[:comp])
	plt.grid(True)
	plt.title('Normalized cumulative sum of eigen values')
	plt.ylabel('Number of Components')
	plt.savefig('report/cumEigenValues.png')
	plt.show()

def showShapeVariations(landmarks, P, b, comp=3):
	meanShape = np.mean(landmarks, axis=0)
	P = np.array(P[:comp]).T; b = np.array(b[:comp])
	f, sp = plt.subplots(comp, 3)
	for i in range(comp):
		for k in [-1.,0.,1.]:
			bNew = [0]*comp
			bNew[i] = k*math.sqrt(b[i])
			shape = meanShape + np.dot(P,bNew)
			sp[i,k+1].plot(shape[::2], shape[1::2])
			sp[i,k+1].set_title('b[{0}] = {1}*sqrt(lamda)'.format(i,int(k)))
			sp[i,k+1].invert_yaxis()
			sp[i,k+1].axes.get_xaxis().set_visible(False)
	plt.savefig('report/shapeVariations.png'.format(i, int(k)))
	plt.show()

def pca(landmarks):
	vals, vecs = eigValsVecs(landmarks)
	#plotEigenValues(vals)
	#showShapeVariations(landmarks, vecs, vals, comp=4)
	return vals, vecs

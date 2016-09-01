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

def plotEigenValues(vals, comp=10, teeth=8, UD=''):
	plt.figure(1)
	plt.plot(range(1,comp+1), vals[:comp])
	plt.grid(True)
	plt.title('Eigenvalues'.format(comp))
	plt.xlabel('Number of Components')
	plt.savefig('report/eigenValues{0}{1}.png'.format(teeth, UD))
	
	cum_val = [0]
	for v in vals:
		cum_val.append(cum_val[-1]+v)
	del cum_val[0]
	cum_val = np.divide(cum_val, cum_val[-1])
	
	plt.figure(2)
	plt.plot(range(1, comp+1), cum_val[:comp])
	plt.grid(True)
	plt.title('Normalized cumulative sum of eigen values')
	plt.xlabel('Number of Components')
	plt.savefig('report/cumEigenValues{0}{1}.png'.format(teeth, UD))
	plt.show()

def showShapeVariations(landmarks, P, b, comp=3, UD=''):
	meanShape = np.mean(landmarks, axis=0)
	P = np.array(P[:comp]).T; b = np.array(b[:comp])
	f, sp = plt.subplots(comp, 3)
	for i in range(comp):
		for k in [-1.,0.,1.]:
			bNew = [0]*comp
			bNew[i] = (3.*k)*math.sqrt(b[i])
			shape = meanShape + np.dot(P,bNew)
			k = int(k)
			sp[i,k+1].plot(shape[::2], shape[1::2])
			sp[i,k+1].set_title('b[{0}] = {1}*sqrt(lamda)'.format(i,int(k)*3))
			sp[i,k+1].invert_yaxis()
			sp[i,k+1].axes.get_xaxis().set_visible(False)
	plt.savefig('report/shapeVariations{0}_{1}{2}.png'.format(comp, len(meanShape)/80, UD))
	plt.show()

def pca(landmarks):
	vals, vecs = eigValsVecs(landmarks)
	#plotEigenValues(vals, teeth=len(landmarks[0])/80, UD='*')
	#showShapeVariations(landmarks, vecs, vals, comp=3, UD='*')
	return vals, vecs

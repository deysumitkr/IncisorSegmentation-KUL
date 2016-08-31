import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np

import alignShape as AS
import pca

def step1(imarr, landmarks, ID=6):
	img = imarr[ID].copy()
	for k in range(0, len(landmarks[ID]), 2):
		cv2.circle(img, (landmarks[ID][k],landmarks[ID][k+1]), 5, (240,230, 0), -1)
	plt.figure(3)
	plt.imshow(img, interpolation = 'bicubic')
	plt.savefig('report/lm_image-{0}.png'.format(ID))
	
	[mx, my, lm] = AS.meanCentering(landmarks)

	plt.figure(1)
	plt.plot(landmarks[ID][::2], landmarks[ID][1::2],'.', label='Actual landmark positions')
	plt.plot(lm[ID][::2], lm[ID][1::2], '.', label='Mean centered landmarks')
	#plt.plot([mx], [my], 'or', label='Mean')
	plt.gca().invert_yaxis()
	plt.grid(True)
	plt.legend()
	plt.savefig('report/lm_centered-{0}.png'.format(ID))

	plt.figure(2)
	nlm = np.divide(lm[ID], np.linalg.norm(lm[ID])) # scale first shape to unit vector
	plt.plot(nlm[::2], nlm[1::2], 'o', label='Normalized landmarks')
	plt.gca().invert_yaxis()
	plt.grid(True)
	plt.legend()
	plt.savefig('report/lm_normalized-{0}.png'.format(ID))
	plt.show()

def step2(landmarks):
	plt.figure(1)
	for i in range(len(landmarks)):
		plt.plot(landmarks[i][::2], landmarks[i][1::2])
	mlm = np.mean(landmarks, axis=0)
	plt.plot(mlm[::2], mlm[1::2], '--k', label='Mean Shape')
	plt.gca().invert_yaxis()
	plt.grid(True)
	plt.legend()
	plt.title('Actual Landmarks')
	plt.savefig('report/allLandmarks.png')
		
	plt.figure(2)
	clandmarks = AS.alignShape(landmarks)
	for i in range(len(clandmarks)):
		plt.plot(clandmarks[i][::2], clandmarks[i][1::2])
	mlm = np.mean(clandmarks, axis=0)
	plt.plot(mlm[::2], mlm[1::2], '--k', label='Mean Shape')
	plt.gca().invert_yaxis()
	plt.grid(True)
	plt.legend()
	plt.title('Aligned Landmarks')
	plt.savefig('report/allAlignedLandmarks.png')

	zaplt.show()

def step3(landmarks):
	ref = np.mean(AS.alignShape(landmarks), axis=0)
	
	for i in range(1,len(landmarks)):
		newLandmarks = [landmarks[i]] + landmarks[:i] + landmarks[(i+1):]
		new = AS.alignShape(landmarks)
		mean = np.mean(new, axis=0)
		plt.plot(mean[::2], mean[1::2], '.')
		print mse(mean, ref)
	plt.plot(ref[::2], ref[1::2], 'o')

	plt.show()


def main(imarr, landmarks):
	ID=6 
	#step1(imarr, landmarks, ID)
	#step2(landmarks[3:6])
	#step3(landmarks)
	#pca.pca(AS.alignShape(landmarks))




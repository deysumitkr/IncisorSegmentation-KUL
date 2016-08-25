import cv2
import numpy as np

def crop(extremes, imgs):
	templates = []
	width = 0; height = 0;
	for i in  range(len(extremes)):
		p1 = extremes[i][0]
		p2 = extremes[i][1]
		x = p1[0]
		y = p1[1]
		w = p2[0]-p1[0]
		h = p2[1]-p1[1]
		templates.append(imgs[i][y:y+h, x:x+w].copy())
		width+=w; height+=h
	width = int(float(width)/len(extremes))
	height = int(float(height)/len(extremes))
	resizeTemplates(templates, width, height)
	return templates, width, height

def resizeTemplates(templates, width, height):
	for i in range(len(templates)):
		templates[i] = cv2.resize(templates[i], (width,height))

def meanTemplate(templates):
	dst = cv2.addWeighted(templates[0],0.5, templates[1],0.5,0)
	for i in range(2,len(templates)):
		dst = cv2.addWeighted(dst,1.-(1./(i+1)), templates[1],1./(i+1),0)
	return dst

def makeTemplates(img, imgs, landmarks):
	extremes = [[(min(l[::2]), min(l[1::2])), (max(l[::2]), max(l[1::2]))] for l in landmarks]
	templates, W, H = crop(extremes, imgs)
	meanTemp = meanTemplate(templates)

	#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	res = cv2.matchTemplate(img, meanTemp, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + W, top_left[1] + H)

	#return [(top_left[0] + int(W/2.0), top_left[1] + int(H*0.25)), (top_left[0] + int(W/2.0), top_left[1] + int(H*0.5)), (top_left[0] + int(W/2.0), top_left[1] + int(H*0.75))]
	return [(top_left[0] + int(W/2.0), top_left[1] + int(H/2.0)), W, H]

	cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)
	cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	cv2.imshow('image', img)
	cv2.waitKey(0)


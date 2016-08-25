import os
import subprocess
import copy
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import alignShape
import pca
import modelFit
import templateMatch
import grayFit

LANDMARKS_PATH = './_Data/Landmarks/original/'
IMAGES_PATH = './_Data/Radiographs/' 

def readLandmarks(img, teeth):
# store landmarks as 2D array
# img is list of image numbers [1..14]
# teeth is list of teeth numbers [1..8]
# each row consists all landmarks from the teeth from a single image
# number of rows = number of images

    mat = []
    for im in img:
        landmarks = []
        for t in teeth:
            path = LANDMARKS_PATH+'landmarks'+str(im)+'-'+str(t)+'.txt'
            f = open(path,'rb')
            landmarks += [int(float(x)) for x in f.read().split('\n') if x]
            f.close()
        mat.append(landmarks)
    return mat

def readImages(images):
    img = []
    for i in images:
        adj = '0' if i<10 else ''
        img.append(cv2.imread(IMAGES_PATH + adj + str(i)+'.tif'))
    return img


def showLandmarks(images, landmarks, minimal = False):
# plot landmarks as circular dots on images
# minimal = True: Show images on openCV window
# minimal = False: Show images on matplotlib window

    for im in range(len(images)):
        adj = '0' if i<10 else ''
        img = cv2.imread(IMAGES_PATH+ adj +str(images[im])+'.tif')
        for i in range(0, len(landmarks[im]), 2):
            cv2.circle(img, (landmarks[im][i],landmarks[im][i+1]), 5, (0,230, 230), -1)

        if minimal:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',img)
            cv2.waitKey(0)
        else:
            plt.imshow(img, interpolation = 'bicubic')
            plt.show()
    

if __name__=='__main__':
    # If not executed from project root, then go a execute from project root
    cwd = os.getcwd()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cwd == dir_path:
        subprocess.call("/bin/bash ../run.sh", shell=True)
        exit()

    # settings
    images = range(1,11)
    teeth = range(5,9)
    
    imarr = readImages(images)
    landmarks = readLandmarks(images,teeth)
    #showLandmarks(images, landmarks, minimal=True)

    testImage = 1
    img = imarr[testImage].copy()
    shape = landmarks[testImage]
    
    location = templateMatch.makeTemplates(img, imarr, readLandmarks(images,range(1,9)))
    meanShape = np.mean(landmarks, axis=0)
    grayFit.shiftLocation(meanShape, (location[0][0], location[0][1]+int(location[2]*0.25)))
    
    
    print "Creating Gray Models..."
    grayModels = grayFit.grayModels(imarr, landmarks, profileLength=20)
    
    print "Creating shape models..."
    clandmarks = alignShape.alignShape(landmarks)
    vals, vecs = pca.pca(clandmarks)
   
    img1 = img.copy()
    for i in range(len(meanShape[::2])):
        cv2.circle(img1, (int(meanShape[::2][i]), int(meanShape[1::2][i])), 14, (0,230, 230), 3)
        cv2.circle(img1, (int(shape[::2][i]), int(shape[1::2][i])), 8, (0,230, 0), 3)
    
    modelShape = copy.copy(meanShape)
   
    for _ in range(2):
        print "Gary Fit..."
        newShape = grayFit.grayFitShape(img, modelShape, grayModels, profileLength=30, metric='min')
        #grayFit.grayFitAll(imarr, landmarks)

        img2 = img.copy()
        for i in range(len(modelShape[::2])):
            cv2.circle(img2, (int(modelShape[::2][i]), int(modelShape[1::2][i])), 12, (0,230, 0), 3)
        for i in range(len(newShape[::2])):
            cv2.circle(img2, (int(newShape[::2][i]), int(newShape[1::2][i])), 8, (250,30, 0), 3)
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img2)
        cv2.waitKey(0)

        print "\nMetric: model-shape " 
        print "Mean squared error: ", mse(shape, modelShape)
        print "Mean absolute error: ", mae(shape, modelShape)
        print

        for _ in range(2):
            #print "Shape fitting..."
            modelShape = np.mean(alignShape.alignShape([modelShape]*2), axis=0)
            modelShape = modelFit.fit(newShape, modelShape, vals, vecs)
            print "Mean squared error: ", mse(shape, modelShape)
        
        print "\nMetric: model-shape " 
        print "Mean squared error: ", mse(shape, modelShape)
        print "Mean absolute error: ", mae(shape, modelShape)
        print

        img2 = img1.copy()
        for i in range(len(modelShape[::2])):
            cv2.circle(img2, (int(modelShape[::2][i]), int(modelShape[1::2][i])), 10, (0,30, 240), 3)
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img2)
        cv2.waitKey(0)




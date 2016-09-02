import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np

import alignShape as AS
import pca
import grayFit as gf
import modelFit
import templateMatch as tm
import preprocess as pp

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

    plt.show()

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

def grayProfileFittings(trainImages, landmarks, testImage, testShape):
    grayModels = gf.grayModels(trainImages, landmarks, profileLength=40)

    pt = 36

    plt.figure()
    plt.plot(grayModels[pt][0], label='Mean Profile')
    plt.title('Mean Gray Profile [point-{0} of top 4 teeth]'.format(pt))
    plt.savefig('report/meanGrayProfile{0}-4U.png'.format(pt))
    #plt.plot(grayModels[1][0])
    #plt.plot(grayModels[20][0])

    plt.figure(2)
    plt.figure(3)
    mins = []
    for k in range(len(testImage)):
        #for point in range(len(testShape[k][::2])):
        for point in [pt]:
            Fgs, line = gf.grayFit(testImage[k], testShape[k], grayModels[point][0], grayModels[point][1], point, profileLength=80)
            grayStrip = gf.getGrayStrip(testImage[k], line)
            plt.figure(2)
            plt.plot(grayStrip, label='Test Image {0}'.format(k+1))
            plt.figure(3)
            plt.plot(Fgs, label='Mahalanobis Distance on Test-{0}'.format(k+1))
            plt.plot(Fgs.index(min(Fgs)), min(Fgs), 'ok')
            mins.append(Fgs.index(min(Fgs)))

    plt.figure(2)
    plt.title('Gray Profile [point-{0} of top 4 teeth]'.format(pt))
    plt.legend()
    plt.savefig('report/grayProfile{0}-4U.png'.format(pt))
    
    plt.figure(3)
    plt.title('Mahalanobis Distances [point-{0} of top 4 teeth]'.format(pt))
    plt.legend()
    plt.savefig('report/md{0}-4U.png'.format(pt))
    
    plt.show()

def grayProfileHist(trainImages, landmarks, testImage, testShape):
    modelProfileLen = 50
    searchProfileLen = 100

    grayModels = gf.grayModels(trainImages, landmarks, profileLength=modelProfileLen)
    
    plt.figure()
    mins = []
    for k in range(len(testImage)):
        for point in range(len(testShape[k][::2])):
            Fgs, line = gf.grayFit(testImage[k], testShape[k], grayModels[point][0], grayModels[point][1], point, profileLength=searchProfileLen)
            grayStrip = gf.getGrayStrip(testImage[k], line)
            plt.plot(Fgs.index(min(Fgs)), min(Fgs), 'ok')
            mins.append(Fgs.index(min(Fgs)))

    plt.xlim([0,len(Fgs)])
    plt.ylabel('Minimum Mahalanobis Distance for each Landmark')
    plt.xlabel('Position on Gray Profile')
    plt.title('Model Profile Length:{0}, Search Profile Length:{1}'.format(modelProfileLen, searchProfileLen))
    plt.savefig('report/mdDistribution-{0}-{1}.png'.format(modelProfileLen, searchProfileLen))

    plt.figure()
    plt.hist(mins)
    plt.ylabel('Frequency of minimum Mahalanobis Distance')
    plt.xlabel('Position on Gray Profile')
    plt.title('Model Profile Length:{0}, Search Profile Length:{1}'.format(modelProfileLen, searchProfileLen))
    plt.savefig('report/mdHist-{0}-{1}.png'.format(modelProfileLen, searchProfileLen))
    plt.show()

def plotFinals(testShape, modelShape, testImg, iterNo, sec):
    plt.figure()
    plt.plot(testShape[::2], testShape[1::2], label='Ground Truth')
    plt.plot(modelShape[::2], modelShape[1::2], label='Model Fit')
    plt.gca().invert_yaxis()
    title = 'Test Image: {0}, Iteration: {1}, MSE: {2:.3f}, MAE: {3:.3f}'.format(testImg, iterNo, mse(modelShape, testShape), mae(modelShape, testShape))
    plt.title(title)
    plt.legend()
    plt.grid(True)
    fname = 'test{0}_{2}_Iter{1}'.format(testImg, iterNo, sec)
    plt.savefig('report/fits/{0}.png'.format(fname))


def finalResults(trainImages, imarr, landmarks, testImage, testShape, testImg, sec):
    location = tm.makeTemplates(testImage, imarr, readLandmarks(trainImages,range(1,9)))
    modelShape = np.mean(landmarks, axis=0)
    if sec == 'U4' or sec == 'T4':
        gf.shiftLocation(modelShape, (location[0][0], location[0][1]-int(location[2]*0.25))) # Upper 4
    elif sec == 'L4' or sec == 'B4':
        gf.shiftLocation(modelShape, (location[0][0], location[0][1]+int(location[2]*0.25))) # Lower 4
    elif sec == '8' or sec == 'A8':
        gf.shiftLocation(modelShape, (location[0][0], location[0][1])) # all 8
    else:
        raise ValueError('Sec not proper. [U4, L4, A8]')
    
    vals, vecs = pca.pca(AS.alignShape(landmarks))
    grayModels = gf.grayModels(imarr, landmarks, profileLength=40)
    print "MSE: ", mse(modelShape, testShape), " MAE: ", mae(modelShape, testShape)
    plotFinals(testShape, modelShape, testImg, 0, sec) 
   
    for k in range(1,6):
        newShape = gf.grayFitShape(testImage, modelShape, grayModels, profileLength=80, metric='min')
        modelShape_tmp = np.mean(AS.alignShape([modelShape]*2), axis=0)
        modelShape = modelFit.fit(newShape, modelShape_tmp, vals, vecs)
        print "MSE: ", mse(modelShape, testShape), " MAE: ", mae(modelShape, testShape)
        plotFinals(testShape, modelShape, testImg, k, sec)
    #plt.show()

def process(images):
    for i in range(len(images)):
        images[i] = pp.process(images[i])
    

def main(imarr, landmarks):
    """
    ID=6
    testImgs =[9]
    teeth = range(1,9)
    trainImages = range(1,15)
    trainImages.remove(testImgs[0])
    print trainImages

    imarr = readImages(trainImages)
    landmarks = readLandmarks(trainImages, teeth)
    testImages = readImages(testImgs)
    testShapes = readLandmarks(testImgs, teeth)

    #process(imarr)
    #process(testImages)
    tm.makeTemplates(testImages[0], imarr, landmarks)

    #step1(imarr, landmarks, ID) # Centering & Normalizing shape
    #step2(landmarks[3:6]) # Procrustes - Before and After Alignment
    #step3(landmarks) # Procrutes - Mean shape invariant of what shape its initialized with
    #AS.alignShape(landmarks, plot=True) # Convergence of shape Alignment
    #pca.pca(AS.alignShape(landmarks))

    #grayProfileFittings(imarr, landmarks, testImages, testShapes)
    #grayProfileHist(imarr, landmarks, testImages, testShapes)
    """
    for x in [1, 2, 9, 11, 12]:
        testImgs = [x]
        for sec in ['U4', 'L4', 'A8']: 
            if sec == 'U4':
                teeth = range(1,5)
            elif sec == 'L4':
                teeth = range(5,9)
            elif sec == 'A8':
                teeth = range(1,9)

            trainImages = range(1,15)
            trainImages.remove(testImgs[0])
            print trainImages

            imarr = readImages(trainImages)
            landmarks = readLandmarks(trainImages, teeth)
            testImages = readImages(testImgs)
            testShapes = readLandmarks(testImgs, teeth)
            finalResults(trainImages, imarr, landmarks, testImages[0], testShapes[0], testImgs[0], sec)

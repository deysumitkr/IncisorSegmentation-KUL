import cv2
import numpy as np

lowThreshold = 0
max_lowThreshold = 160
ratio = 3
kernel_size = 3 

def process(img):
    #img = img1.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.GaussianBlur(img,(5,5),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    #img = cv2.medianBlur(img, 3)
    #result = cv2.Sobel(img,6,0,1,ksize=5); img = cv2.medianBlur(cv2.convertScaleAbs(result),3)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
    cv2.normalize(img,img,0,255,cv2.NORM_MINMAX);
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    return img


def CannyThreshold(lowThreshold):
    global gray, img
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',detected_edges)

def canny(im):
    global gray, img
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(gray)
    #img = cv2.equalizeHist(gray)

    cv2.namedWindow('canny demo', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

    CannyThreshold(0)  # initialization
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

import cv2
import numpy as np

def FAST_feature(img):

    #Can be used with SIFT, SURF, BRISK, BRIEF, ORB or FREAK as descriptor extractor

    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp


def STAR_feature(img):

    #Can be used with SIFT, SURF, BRISK, BRIEF, ORB or FREAK as descriptor extractor

    fast = cv2.xfeatures2d.StarDetector_create()
    kp = fast.detect(img, None)
    return kp


def SIFT_feature(img):

    #Can be used with SURF, BRISK, BRIEF, ORB or FREAK as descriptor extractor
    #Do not work with SIFT maybe because of an issue (https://github.com/opencv/opencv/pull/6078)

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    return kp


def SURF_feature(img):

    #Can be used with SIFT, SURF, BRIEF, BRISK, ORB and FREAK as descriptor extractor

    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400, upright=False)
    kp = surf.detect(img, None)

    return kp

def ORB_feature(img):

    #Can be used with SIFT, SURF, BRIEF, BRISK, ORB and FREAK as descriptor extractor

    orb = cv2.ORB_create(nfeatures=5000, edgeThreshold=1, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0)
    kp = orb.detect(img,None)
    return kp

def BRISK_feature(img):

    #Can be used with SIFT, SURF, BRIEF, BRISK, ORB and FREAK as descriptor extractor

    brisk = cv2.BRISK_create()
    kp = brisk.detect(img, None)

    return kp

def SIMPLEBLOB_feature(img):

    #Can be used with SIFT, SURF, BRIEF, BRISK, ORB and FREAK as descriptor extractor

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 2000;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    simbpleblob = cv2.SimpleBlobDetector_create()
    kp = simbpleblob.detect(img)

    return kp

import cv2
import numpy as np

def FAST_feature(img):

    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    return kp


def STAR_feature(img):

    fast = cv2.xfeatures2d.StarDetector_create()
    kp = fast.detect(img, None)
    return kp


def SIFT_feature(img):

    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    return kp


def SURF_feature(img):

    surf = cv2.xfeatures2d.SURF_create()
    kp = surf.detect(img, None)

    return kp

def ORB_feature(img):

    orb = cv2.ORB_create()
    kp = orb.detect(img,None)
    return kp

def BRISK_feature(img):

    brisk = cv2.BRISK_create()
    kp = brisk.detect(img, None)

    return kp

def MSER_feature(img):

    mser = cv2.MSER_create()
    kp = mser.detect(img, None)

    return kp

def SIMPLEBLOB_feature(img):

    simbpleblob = cv2.SimpleBlobDetector_create()
    kp = simbpleblob.detect(img)

    return kp

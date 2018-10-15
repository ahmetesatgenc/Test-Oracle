import cv2

# kp shall be output of the .compute function. Since, it converts the type of the "des" from tuple to ndarray

def SIFT_desc_extract(img, kp):

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.compute(img, kp)
    return kp, des


def SURF_desc_extract(img,kp):

    surf = cv2.xfeatures2d.SIFT_create()
    kp, des = surf.compute(img, kp)
    return kp, des


def BRIEF_desc_extract(img,kp):

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(img, kp)
    return kp, des


def BRISK_desc_extract(img,kp):

    brisk = cv2.BRISK_create()
    kp, des = brisk.compute(img, kp)
    return kp, des


def ORB_desc_extract(img,kp):

    # It seems that there is a bug related to ORB.detectandcompute (https://github.com/opencv/opencv/pull/6078)
    #Gives this error "error: (-215) u != 0 in function cv::Mat::create" when SIFT is used as feature detector.

    orb = cv2.ORB_create()
    kp, des = orb.compute(img, kp)
    return kp, des


def FREAK_desc_extract(img,kp):
    freak = cv2.xfeatures2d.FREAK_create()
    kp, des = freak.compute(img, kp)
    return kp, des

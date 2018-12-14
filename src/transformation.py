import cv2
import numpy as np

def Affine_transform(kp1, kp2, img_grab, goodMatches):

    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches])
    sz = img_grab.shape
    warp_matrix = cv2.estimateRigidTransform(src_pts, dst_pts, fullAffine=True)
    if warp_matrix is not None:
        warpedImage = cv2.warpAffine(np.asarray(img_grab, dtype=np.float32), warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        warpedImage = img_grab

    return np.asarray(warpedImage, dtype=np.uint8)


def Perspective_transform(kp1,kp2,img_grab,goodMatches):

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,6.0)
    matchesMask = mask.ravel().tolist()
    goodMatches = [m for (i, m) in enumerate(goodMatches) if matchesMask[i] == 1]

    h,w, _= img_grab.shape

    if M is not None:
        warpedImage = cv2.warpPerspective(img_grab,M,(w,h))
    else:
        warpedImage = img_grab
    return warpedImage
import cv2
import numpy as np

methods_hist = cv2.HISTCMP_CORREL
# Other available methods'cv2.HISTCMP_BHATTACHARYYA', 'cv2.HISTCMP_CHISQR_ALT', 'cv2.HISTCMP_INTERSECT', 'cv2.HISTCMP_CHISQR', 'cv2.HISTCMP_HELLINGER', 'cv2.HISTCMP_KL_DIV'

methods_template = cv2.TM_CCOEFF_NORMED
# Other available methods 'cv2.TM_CCORR','cv2.TM_CCOEFF_NORMED','cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'

#window_sizes = [(640, 360), (480, 270), (384, 216), (320, 180), (240, 135), (192, 108), (160, 108), (120, 90), (96, 90),
#                (64, 60), (48, 45), (32, 30), (16, 20), (16, 15)]

window_sizes = [(640, 360), (480, 270), (320, 180), (160, 108), (120, 90),(16, 20)]

def getHist(img):

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cv2.normalize(hist, 0, 255, cv2.NORM_L2)

    return hist


def Histogram_matching(img1, img2):
    hist_warped = getHist(img2)
    hist_ref = getHist(img1)
    hist_matching = cv2.compareHist(hist_warped, hist_ref, methods_hist)
    print "histogram_matching", hist_matching
    return hist_matching

def Pixel_matching(img1, img2):

    pixel_result = []
    wgrab, hgrab, _ = img1.shape
    pixel_comp_0 = np.sum((img2[:, :, 0] - img1[:, :, 0]) ** 2) / (wgrab * hgrab)
    pixel_comp_1 = np.sum((img2[:, :, 1] - img1[:, :, 1]) ** 2) / (wgrab * hgrab)
    pixel_comp_2 = np.sum((img2[:, :, 2] - img1[:, :, 2]) ** 2) / (wgrab * hgrab)
    # print "pixel_comp_0 =", pixel_comp_0
    # print "pixel_comp_1 =", pixel_comp_1
    # print "pixel_comp_2 =", pixel_comp_2
    pixel_result.append(pixel_comp_0)
    pixel_result.append(pixel_comp_1)
    pixel_result.append(pixel_comp_2)
    print "pixel_matching", max(pixel_result)

    return np.max(pixel_result)

def Template_Matching(img1, img2):
    max_val_array_all_windows = []
    wgrab, hgrab, _ = img2.shape
    for new_window in range(0, len(window_sizes)):
        window_w, window_h = window_sizes[new_window]
        for i in range(0, wgrab, window_w):
            for j in range(0, hgrab, window_h):
                max_val_array = []
                template_2 = img2[i:i + window_w, j:j + window_h]
                template_1 = img1[i:i + window_w, j:j + window_h]
                img_grab = template_2
                img_ref = template_1

                res = cv2.matchTemplate(img_grab, img_ref, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val == 0.0:
                    res = cv2.matchTemplate(img_ref, img_grab, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # print "max_val", max_val
                max_val_array.append(max_val)
        max_val_array_all_windows.append(np.min(max_val_array))

    print "min temp match=", np.min(max_val_array_all_windows)
    return (max_val_array_all_windows)
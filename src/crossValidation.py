import numpy as np
import os
from TestOracleConfMainv2 import kf
from imageMatching import window_sizes


def crossValidation(hist_test,pixel_test,template_test,folder,max_index_1_av,max_index_2_av,max_index_05_av,result_folder,hist_enable,pix_enable,temp_enable):
    if temp_enable is False:
        length_of_windows= 1
    else:
        length_of_windows = len(window_sizes)
    fn_array_2_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_1_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_3_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_4_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_1_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_3_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_4_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tp_array_2_f1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fn_array_2_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_1_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_3_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_4_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_1_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_3_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_4_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tp_array_2_f2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fn_array_2_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_1_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_3_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    fp_array_4_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_1_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_3_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tn_array_4_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]
    tp_array_2_f05 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(length_of_windows)]

    for infolder in range (0,folder,1):
        for i in range (0,len(max_index_1_av),1):
            for j in range (0,kf.n_splits,1):
                fp_f1=0;tn_f1=0;fn_f1=0;tp_f1=0;fp_f2=0;tn_f2=0;fn_f2=0;tp_f2=0;fp_f05=0;tn_f05=0;fn_f05=0;tp_f05=0
                if len(hist_test) is not 0:
                    split_range = (min([len(x) for x in (hist_test[infolder])]))
                elif len(pixel_test) is not 0:
                    split_range = (min([len(x) for x in (pixel_test[infolder])]))
                else:
                    split_range = (min([len(x) for x in (template_test[infolder])]))
                for k in range(0, split_range, 1):
                    if hist_enable is True and (pix_enable is False and temp_enable is False):
                        if infolder+1 == 1:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (hist_test[infolder][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (hist_test[infolder][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (hist_test[infolder][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    elif pix_enable is True and (hist_enable is False and temp_enable is False):
                        if infolder+1 == 1:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (pixel_test[infolder][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (pixel_test[infolder][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (pixel_test[infolder][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    elif temp_enable is True and (hist_enable is False and pix_enable is False):
                        if infolder+1 == 1:
                            if (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (template_test[infolder][0][i][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (template_test[infolder][0][i][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (template_test[infolder][0][i][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    elif (hist_enable is True and pix_enable is True) and temp_enable is False:
                        if infolder+1 == 1:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (hist_test[infolder][j][k] > max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (hist_test[infolder][j][k] > max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (hist_test[infolder][j][k] > max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    elif (pix_enable is True and temp_enable is True) and hist_enable is False:
                        if infolder+1 == 1:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (pixel_test[infolder][j][k] > max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (pixel_test[infolder][j][k] > max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (pixel_test[infolder][j][k] > max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    elif (hist_enable is True and temp_enable is True) and pix_enable is False:
                        if infolder+1 == 1:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (hist_test[infolder][j][k] > max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (hist_test[infolder][j][k] > max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (hist_test[infolder][j][k] > max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                    else:
                        if infolder+1 == 1:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 2:
                            if (hist_test[infolder][j][k] > max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_1_av[i] / 100):
                                fn_f1 = fn_f1 + 1
                            else:
                                tp_f1 = tp_f1 + 1
                            if (hist_test[infolder][j][k] > max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_2_av[i] / 100):
                                fn_f2 = fn_f2 + 1
                            else:
                                tp_f2 = tp_f2 + 1
                            if (hist_test[infolder][j][k] > max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] > max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] > max_index_05_av[i] / 100):
                                fn_f05 = fn_f05 + 1
                            else:
                                tp_f05 = tp_f05 + 1
                        if infolder+1 == 3:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1
                        if infolder+1 == 4:
                            if (hist_test[infolder][j][k] < max_index_1_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_1_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_1_av[i] / 100):
                                fp_f1 = fp_f1 + 1
                            else:
                                tn_f1 = tn_f1 + 1
                            if (hist_test[infolder][j][k] < max_index_2_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_2_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_2_av[i] / 100):
                                fp_f2 = fp_f2 + 1
                            else:
                                tn_f2 = tn_f2 + 1
                            if (hist_test[infolder][j][k] < max_index_05_av[i] / 100) and (pixel_test[infolder][j][k] < max_index_05_av[i] / 100) and (template_test[infolder][0][i][j][k] < max_index_05_av[i] / 100):
                                fp_f05 = fp_f05 + 1
                            else:
                                tn_f05 = tn_f05 + 1

                if infolder+1 == 1:
                    fp_array_1_f1[i][j].append(fp_f1)
                    tn_array_1_f1[i][j].append(tn_f1)
                    fp_array_1_f2[i][j].append(fp_f2)
                    tn_array_1_f2[i][j].append(tn_f2)
                    fp_array_1_f05[i][j].append(fp_f05)
                    tn_array_1_f05[i][j].append(tn_f05)
                if infolder+1 == 2:
                    fn_array_2_f1[i][j].append(fn_f1)
                    tp_array_2_f1[i][j].append(tp_f1)
                    fn_array_2_f2[i][j].append(fn_f2)
                    tp_array_2_f2[i][j].append(tp_f2)
                    fn_array_2_f05[i][j].append(fn_f05)
                    tp_array_2_f05[i][j].append(tp_f05)
                if infolder+1 == 3:
                    fp_array_3_f1[i][j].append(fp_f1)
                    tn_array_3_f1[i][j].append(tn_f1)
                    fp_array_3_f2[i][j].append(fp_f2)
                    tn_array_3_f2[i][j].append(tn_f2)
                    fp_array_3_f05[i][j].append(fp_f05)
                    tn_array_3_f05[i][j].append(tn_f05)
                if infolder+1 == 4:
                    fp_array_4_f1[i][j].append(fp_f1)
                    tn_array_4_f1[i][j].append(tn_f1)
                    fp_array_4_f2[i][j].append(fp_f2)
                    tn_array_4_f2[i][j].append(tn_f2)
                    fp_array_4_f05[i][j].append(fp_f05)
                    tn_array_4_f05[i][j].append(tn_f05)

    accuracy_array_f1 = accuracy(fp_array_1_f1,tn_array_1_f1,fn_array_2_f1,tp_array_2_f1,fp_array_3_f1,tn_array_3_f1,fp_array_4_f1,tn_array_4_f1,length_of_windows)
    accuracy_array_f2 = accuracy(fp_array_1_f2,tn_array_1_f2,fn_array_2_f2,tp_array_2_f2,fp_array_3_f2,tn_array_3_f2,fp_array_4_f2,tn_array_4_f2,length_of_windows)
    accuracy_array_f05 = accuracy(fp_array_1_f05,tn_array_1_f05,fn_array_2_f05,tp_array_2_f05,fp_array_3_f05,tn_array_3_f05,fp_array_4_f05,tn_array_4_f05,length_of_windows)
    if not os.path.exists(result_folder + 'crossvalidation' + '.txt'):
        with open(result_folder + 'crossvalidation' + '.txt', "w") as file:
            file.close()
    with open(result_folder + 'crossvalidation' + '.txt', "a") as file:
        for k in range(0, length_of_windows, 1):
            file.write("Window" + str(window_sizes[k]) + "\n")
            file.write("fp_f1 = " + str(mean(fp_array_1_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f1,length_of_windows)[k][0]+mean(tn_array_1_f1,length_of_windows)[k][0]) + "\n")
            file.write("tn_f1 = " + str(mean(tn_array_1_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f1,length_of_windows)[k][0]+mean(tn_array_1_f1,length_of_windows)[k][0]) + "\n")
            file.write("fn_f1 = " + str(mean(fn_array_2_f1,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f1,length_of_windows)[k][0]+mean(tp_array_2_f1,length_of_windows)[k][0]) + "\n")
            file.write("tp_f1 = " + str(mean(tp_array_2_f1,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f1,length_of_windows)[k][0]+mean(tp_array_2_f1,length_of_windows)[k][0]) + "\n")
            file.write("fp_f1 = " + str(mean(fp_array_3_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f1,length_of_windows)[k][0]+mean(tn_array_3_f1,length_of_windows)[k][0]) + "\n")
            file.write("tn_f1 = " + str(mean(tn_array_3_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f1,length_of_windows)[k][0]+mean(tn_array_3_f1,length_of_windows)[k][0]) + "\n")
            file.write("fp_f1 = " + str(mean(fp_array_4_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f1,length_of_windows)[k][0]+mean(tn_array_4_f1,length_of_windows)[k][0]) + "\n")
            file.write("tn_f1 = " + str(mean(tn_array_4_f1,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f1,length_of_windows)[k][0]+mean(tn_array_4_f1,length_of_windows)[k][0]) + "\n")
            file.write("accuracy = " + str(accuracy_array_f1[k]) + "\n")
            file.write("threshold = " + str(max_index_1_av[k]) + "\n")
            file.write("\n")
            file.write("fp_f2 = " + str(mean(fp_array_1_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f2,length_of_windows)[k][0]+mean(tn_array_1_f2,length_of_windows)[k][0]) + "\n")
            file.write("tn_f2 = " + str(mean(tn_array_1_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f2,length_of_windows)[k][0]+mean(tn_array_1_f2,length_of_windows)[k][0]) + "\n")
            file.write("fn_f2 = " + str(mean(fn_array_2_f2,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f2,length_of_windows)[k][0]+mean(tp_array_2_f2,length_of_windows)[k][0]) + "\n")
            file.write("tp_f2 = " + str(mean(tp_array_2_f2,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f2,length_of_windows)[k][0]+mean(tp_array_2_f2,length_of_windows)[k][0]) + "\n")
            file.write("fp_f2 = " + str(mean(fp_array_3_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f2,length_of_windows)[k][0]+mean(tn_array_3_f2,length_of_windows)[k][0]) + "\n")
            file.write("tn_f2 = " + str(mean(tn_array_3_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f2,length_of_windows)[k][0]+mean(tn_array_3_f2,length_of_windows)[k][0]) + "\n")
            file.write("fp_f2 = " + str(mean(fp_array_4_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f2,length_of_windows)[k][0]+mean(tn_array_4_f2,length_of_windows)[k][0]) + "\n")
            file.write("tn_f2 = " + str(mean(tn_array_4_f2,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f2,length_of_windows)[k][0]+mean(tn_array_4_f2,length_of_windows)[k][0]) + "\n")
            file.write("accuracy = " + str(accuracy_array_f2[k]) + "\n")
            file.write("threshold = " + str(max_index_2_av[k]) + "\n")
            file.write("\n")
            file.write("fp_f05 = " + str(mean(fp_array_1_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f05,length_of_windows)[k][0]+mean(tn_array_1_f05,length_of_windows)[k][0]) + "\n")
            file.write("tn_f05 = " + str(mean(tn_array_1_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_1_f05,length_of_windows)[k][0]+mean(tn_array_1_f05,length_of_windows)[k][0]) + "\n")
            file.write("fn_f05 = " + str(mean(fn_array_2_f05,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f05,length_of_windows)[k][0]+mean(tp_array_2_f05,length_of_windows)[k][0]) + "\n")
            file.write("tp_f05 = " + str(mean(tp_array_2_f05,length_of_windows)[k][0]) + '/' + str(mean(fn_array_2_f05,length_of_windows)[k][0]+mean(tp_array_2_f05,length_of_windows)[k][0]) + "\n")
            file.write("fp_f05 = " + str(mean(fp_array_3_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f05,length_of_windows)[k][0]+mean(tn_array_3_f05,length_of_windows)[k][0]) + "\n")
            file.write("tn_f05 = " + str(mean(tn_array_3_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_3_f05,length_of_windows)[k][0]+mean(tn_array_3_f05,length_of_windows)[k][0]) + "\n")
            file.write("fp_f05 = " + str(mean(fp_array_4_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f05,length_of_windows)[k][0]+mean(tn_array_4_f05,length_of_windows)[k][0]) + "\n")
            file.write("tn_f05 = " + str(mean(tn_array_4_f05,length_of_windows)[k][0]) + '/' + str(mean(fp_array_4_f05,length_of_windows)[k][0]+mean(tn_array_4_f05,length_of_windows)[k][0]) + "\n")
            file.write("accuracy = " + str(accuracy_array_f05[k]) + "\n")
            file.write("threshold = " + str(max_index_05_av[k]) + "\n")
        file.close()

    return mean(fp_array_1_f1,length_of_windows), mean(tn_array_1_f1,length_of_windows), mean(fn_array_2_f1,length_of_windows), mean(tp_array_2_f1,length_of_windows), mean(fp_array_3_f1,length_of_windows), mean(tn_array_3_f1,length_of_windows), mean(fp_array_4_f1,length_of_windows), mean(tn_array_4_f2,length_of_windows),mean(fp_array_1_f2,length_of_windows), mean(tn_array_1_f2,length_of_windows), mean(fn_array_2_f2,length_of_windows), mean(tp_array_2_f2,length_of_windows), mean(fp_array_3_f2,length_of_windows), mean(tn_array_3_f2,length_of_windows), mean(fp_array_4_f2,length_of_windows), mean(tn_array_4_f2,length_of_windows),mean(fp_array_1_f05,length_of_windows), mean(tn_array_1_f05,length_of_windows), mean(fn_array_2_f05,length_of_windows), mean(tp_array_2_f05,length_of_windows), mean(fp_array_3_f05,length_of_windows), mean(tn_array_3_f05,length_of_windows), mean(fp_array_4_f05,length_of_windows), mean(tn_array_4_f05,length_of_windows)

def mean(array,length_of_windows):
    array_cv = [[] * 1 for i in range(length_of_windows)]
    for i in range(length_of_windows):
        temp = []
        for j in range(kf.n_splits):
            temp.append(array[i][j][0])
        mean = float(sum(temp)) / len(temp)
        array_cv[i].append(mean)
    return array_cv

def accuracy(fp_1, tn_1, fn_2,tp_2,fp_3,tn_3,fp_4,tn_4,length_of_windows):
    accuracy_array = [[] * 1 for i in range(length_of_windows)]
    for i in range(0, length_of_windows, 1):
        accuracy_array[i] = float(mean(tn_1,length_of_windows)[i][0]+mean(tp_2,length_of_windows)[i][0]+mean(tn_3,length_of_windows)[i][0]+mean(tn_4,length_of_windows)[i][0])/ float(
            mean(tn_1,length_of_windows)[i][0] + mean(fp_1,length_of_windows)[i][0]+mean(tp_2,length_of_windows)[i][0]+mean(fn_2,length_of_windows)[i][0] + mean(tn_3,length_of_windows)[i][0]+mean(fp_3,length_of_windows)[i][0] + mean(tn_4,length_of_windows)[i][0]+mean(fp_4,length_of_windows)[i][0])
    return  accuracy_array




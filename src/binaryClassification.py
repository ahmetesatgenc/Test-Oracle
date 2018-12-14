import numpy as np
from imageMatching import window_sizes
from TestOracleConfMainv2 import kf


fn_array_2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
fp_array_1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
fp_array_3 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
fp_array_4 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tn_array_1 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tn_array_3 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tn_array_4 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tp_array_2 = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]

fp_sat_array =  [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tn_sat_array =  [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
fn_fail_array = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]
tp_fail_array = [[[] * 1 for i in range(kf.n_splits)] * 1 for i in range(len(window_sizes))]



def one_IM(array_one, selected_dataset, folder,splits):
    if selected_dataset == 1:
        if type(array_one[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    # print "threshold = ", threshold
                    for j in range(0, len(window_sizes), 1):
                        fp = 0;
                        tn = 0;
                        fn = 0;
                        tp = 0;
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 2:
                                if (array_one[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 1:
                                if (array_one[j][k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                        if folder == 2:
                            fp_sat_array[j][k].append(fp)
                            tn_sat_array[j][k].append(tn)
                        if folder == 1:
                            fn_fail_array[j][k].append(fn)
                            tp_fail_array[j][k].append(tp)
        else:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    # print "threshold = ", threshold
                    fp = 0;
                    tn = 0;
                    fn = 0;
                    tp = 0;
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 2:
                            if (array_one[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1


                        elif folder == 1:
                            if (array_one[k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                    if folder == 2:
                        fp_sat_array[0][k].append(fp)
                        tn_sat_array[0][k].append(tn)
                    if folder == 1:
                        fn_fail_array[0][k].append(fn)
                        tp_fail_array[0][k].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        if type(array_one[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    for j in range(0, len(window_sizes), 1):
                        fn = 0;
                        fp = 0;
                        tp = 0;
                        tn = 0
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 1:
                                if (array_one[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 2:
                                if (array_one[j][k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                            elif folder == 3:
                                if (array_one[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 4:
                                if (array_one[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                        if folder == 1:
                            fp_array_1[j][k].append(fp)
                            tn_array_1[j][k].append(tn)
                        if folder == 2:
                            fn_array_2[j][k].append(fn)
                            tp_array_2[j][k].append(tp)
                        if folder == 3:
                            fp_array_3[j][k].append(fp)
                            tn_array_3[j][k].append(tn)
                        if folder == 4:
                            fp_array_4[j][k].append(fp)
                            tn_array_4[j][k].append(tn)
        else:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):

                    fn = 0;
                    fp = 0;
                    tp = 0;
                    tn = 0
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 1:
                            if (array_one[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[0][k].append(fp)
                        tn_array_1[0][k].append(tn)
                    if folder == 2:
                        fn_array_2[0][k].append(fn)
                        tp_array_2[0][k].append(tp)
                    if folder == 3:
                        fp_array_3[0][k].append(fp)
                        tn_array_3[0][k].append(tn)
                    if folder == 4:
                        fp_array_4[0][k].append(fp)
                        tn_array_4[0][k].append(tn)

    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


def two_IM(array_one, array_two, selected_dataset, folder,splits):
    if selected_dataset == 1:
        if type(array_one[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    for j in range(0, len(window_sizes), 1):
                        fn = 0
                        fp = 0
                        tp = 0
                        tn = 0
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 2:
                                if (array_one[j][k][i] < threshold) and (array_two[k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 1:
                                if (array_one[j][k][i] > threshold) and (array_two[k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                        if folder == 2:
                            fp_sat_array[k][k].append(fp)
                            tn_sat_array[j][k].append(tn)
                        if folder == 1:
                            fn_fail_array[j][k].append(fn)
                            tp_fail_array[j][k].append(tp)

        elif type(array_two[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    for j in range(0, len(window_sizes), 1):
                        fn = 0
                        fp = 0
                        tp = 0
                        tn = 0
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 2:
                                if (array_one[k][i] < threshold) and (array_two[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 1:
                                if (array_one[k][i] > threshold) and (array_two[j][k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                        if folder == 2:
                            fp_sat_array[k][j].append(fp)
                            tn_sat_array[j][k].append(tn)
                        if folder == 1:
                            fn_fail_array[j][k].append(fn)
                            tp_fail_array[j][k].append(tp)
        else:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 2:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 1:
                            if (array_one[k][i] > threshold) and (array_two[k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1
                    if folder == 2:
                        fp_sat_array[0][k].append(fp)
                        tn_sat_array[0][k].append(tn)
                    if folder == 1:
                        fn_fail_array[0][k].append(fn)
                        tp_fail_array[0][k].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        if type(array_one[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    for j in range(0, len(window_sizes), 1):
                        fn = 0
                        fp = 0
                        tp = 0
                        tn = 0
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 1:
                                if (array_one[j][k][i] < threshold) and (array_two[k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 2:
                                if (array_one[j][k][i] > threshold) and (array_two[k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                            elif folder == 3:
                                if (array_one[j][k][i] < threshold) and (array_two[k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 4:
                                if (array_one[j][k][i] < threshold) and (array_two[k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                        if folder == 1:
                            fp_array_1[j][k].append(fp)
                            tn_array_1[j][k].append(tn)
                        if folder == 2:
                            fn_array_2[j][k].append(fn)
                            tp_array_2[j][k].append(tp)
                        if folder == 3:
                            fp_array_3[j][k].append(fp)
                            tn_array_3[j][k].append(tn)
                        if folder == 4:
                            fp_array_4[j][k].append(fp)
                            tn_array_4[j][k].append(tn)
        elif type(array_two[0][0]) is list:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    for j in range(0, len(window_sizes), 1):
                        fn = 0
                        fp = 0
                        tp = 0
                        tn = 0
                        for i in range(0, (min([len(x) for x in array_one])), 1):
                            if folder == 1:
                                if (array_one[k][i] < threshold) and (array_two[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 2:
                                if (array_one[k][i] > threshold) and (array_two[j][k][i] > threshold):
                                    fn = fn + 1
                                else:
                                    tp = tp + 1

                            elif folder == 3:
                                if (array_one[k][i] < threshold) and (array_two[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                            elif folder == 4:
                                if (array_one[k][i] < threshold) and (array_two[j][k][i] < threshold):
                                    fp = fp + 1
                                else:
                                    tn = tn + 1

                        if folder == 1:
                            fp_array_1[j][k].append(fp)
                            tn_array_1[j][k].append(tn)
                        if folder == 2:
                            fn_array_2[j][k].append(fn)
                            tp_array_2[j][k].append(tp)
                        if folder == 3:
                            fp_array_3[j][k].append(fp)
                            tn_array_3[j][k].append(tn)
                        if folder == 4:
                            fp_array_4[j][k].append(fp)
                            tn_array_4[j][k].append(tn)
        else:
            for k in range(0, splits, 1):
                for threshold in np.arange(0.0, 1.01, 0.01):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 1:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[k][i] > threshold) and (array_two[k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[0][k].append(fp)
                        tn_array_1[0][k].append(tn)
                    if folder == 2:
                        fn_array_2[0][k].append(fn)
                        tp_array_2[0][k].append(tp)
                    if folder == 3:
                        fp_array_3[0][k].append(fp)
                        tn_array_3[0][k].append(tn)
                    if folder == 4:
                        fp_array_4[0][k].append(fp)
                        tn_array_4[0][k].append(tn)

    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


def all_IM(array_one, array_two, array_three, selected_dataset, folder,splits):
    if selected_dataset == 1:
        for k in range(0, splits, 1):
            for threshold in np.arange(0.0, 1.01, 0.01):
                # print "threshold = ", threshold
                for j in range(0, len(window_sizes), 1):
                    fp = 0;
                    tn = 0;
                    fn = 0;
                    tp = 0;
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 2:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold) and (
                                    array_three[j][k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 1:
                            if (array_one[k][i] > threshold) and (array_two[k][i] > threshold) and (
                                    array_three[j][k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                    if folder == 2:
                        fp_sat_array[j][k].append(fp)
                        tn_sat_array[j][k].append(tn)
                    if folder == 1:
                        fn_fail_array[j][k].append(fn)
                        tp_fail_array[j][k].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        for k in range(0, splits, 1):
            for threshold in np.arange(0.0, 1.01, 0.01):
                for j in range(0, len(window_sizes), 1):
                    fn = 0;
                    fp = 0;
                    tp = 0;
                    tn = 0
                    # print "threshold = ", threshold
                    for i in range(0, (min([len(x) for x in array_one])), 1):
                        if folder == 1:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold) and (
                                    array_three[j][k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[k][i] > threshold) and (array_two[k][i] > threshold) and (
                                    array_three[j][k][i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold) and (
                                    array_three[j][k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[k][i] < threshold) and (array_two[k][i] < threshold) and (
                                    array_three[j][k][i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[j][k].append(fp)
                        tn_array_1[j][k].append(tn)
                    if folder == 2:
                        fn_array_2[j][k].append(fn)
                        tp_array_2[j][k].append(tp)
                    if folder == 3:
                        fp_array_3[j][k].append(fp)
                        tn_array_3[j][k].append(tn)
                    if folder == 4:
                        fp_array_4[j][k].append(fp)
                        tn_array_4[j][k].append(tn)



    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


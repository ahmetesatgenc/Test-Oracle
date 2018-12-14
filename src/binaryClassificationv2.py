import numpy as np
from imageMatching import window_sizes
from TestOracleConfMainv2 import kf


fn_array_2 = [[] * 101 for i in range(len(window_sizes))]
fp_array_1 = [[] * 101 for i in range(len(window_sizes))]
fp_array_3 = [[] * 101 for i in range(len(window_sizes))]
fp_array_4 = [[] * 101 for i in range(len(window_sizes))]
tn_array_1 = [[] * 101 for i in range(len(window_sizes))]
tn_array_3 = [[] * 101 for i in range(len(window_sizes))]
tn_array_4 = [[] * 101 for i in range(len(window_sizes))]
tp_array_2 = [[] * 101 for i in range(len(window_sizes))]

fp_sat_array = [[] * 101 for i in range(len(window_sizes))]
tn_sat_array = [[] * 101 for i in range(len(window_sizes))]
fn_fail_array = [[] * 101 for i in range(len(window_sizes))]
tp_fail_array = [[] * 101 for i in range(len(window_sizes))]



def one_IM(array_one, selected_dataset, folder,splits):
    if selected_dataset == 1:
        if type(array_one[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):
                # print "threshold = ", threshold
                for j in range(0, len(window_sizes), 1):
                    fp = 0;
                    tn = 0;
                    fn = 0;
                    tp = 0;
                    for i in range(0, len(array_one), 1):
                        if folder == 2:
                            if (array_one[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 1:
                            if (array_one[i][j] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                    if folder == 2:
                        fp_sat_array[j].append(fp)
                        tn_sat_array[j].append(tn)
                    if folder == 1:
                        fn_fail_array[j].append(fn)
                        tp_fail_array[j].append(tp)
        else:
            for threshold in np.arange(0.0, 1.01, 0.01):
                # print "threshold = ", threshold
                fp = 0;
                tn = 0;
                fn = 0;
                tp = 0;
                for i in range(0, len(array_one), 1):
                    if folder == 2:
                        if (array_one[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1


                    elif folder == 1:
                        if (array_one[i] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1

                if folder == 2:
                    fp_sat_array[0].append(fp)
                    tn_sat_array[0].append(tn)
                if folder == 1:
                    fn_fail_array[0].append(fn)
                    tp_fail_array[0].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        if type(array_one[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):

                for j in range(0, len(window_sizes), 1):
                    fn = 0;
                    fp = 0;
                    tp = 0;
                    tn = 0

                    # print "threshold = ", threshold
                    for i in range(0, len(array_one), 1):
                        if folder == 1:
                            if (array_one[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[i][j] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[j].append(fp)
                        tn_array_1[j].append(tn)
                    if folder == 2:
                        fn_array_2[j].append(fn)
                        tp_array_2[j].append(tp)
                    if folder == 3:
                        fp_array_3[j].append(fp)
                        tn_array_3[j].append(tn)
                    if folder == 4:
                        fp_array_4[j].append(fp)
                        tn_array_4[j].append(tn)
        else:
            for threshold in np.arange(0.0, 1.01, 0.01):

                fn = 0;
                fp = 0;
                tp = 0;
                tn = 0

                # print "threshold = ", threshold
                for i in range(0, len(array_one), 1):
                    if folder == 1:
                        if (array_one[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 2:
                        if (array_one[i] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1

                    elif folder == 3:
                        if (array_one[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 4:
                        if (array_one[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                if folder == 1:
                    fp_array_1[0].append(fp)
                    tn_array_1[0].append(tn)
                if folder == 2:
                    fn_array_2[0].append(fn)
                    tp_array_2[0].append(tp)
                if folder == 3:
                    fp_array_3[0].append(fp)
                    tn_array_3[0].append(tn)
                if folder == 4:
                    fp_array_4[0].append(fp)
                    tn_array_4[0].append(tn)

    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


def two_IM(array_one, array_two, selected_dataset, folder,splits):
    if selected_dataset == 1:
        if type(array_one[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):
                for j in range(0, len(window_sizes), 1):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, len(array_one), 1):
                        if folder == 2:
                            if (array_one[i][j] < threshold) and (array_two[i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 1:
                            if (array_one[i][j] > threshold) and (array_two[i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                    if folder == 2:
                        fp_sat_array[j].append(fp)
                        tn_sat_array[j].append(tn)
                    if folder == 1:
                        fn_fail_array[j].append(fn)
                        tp_fail_array[j].append(tp)

        elif type(array_two[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):
                for j in range(0, len(window_sizes), 1):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, len(array_one), 1):
                        if folder == 2:
                            if (array_one[i] < threshold) and (array_two[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 1:
                            if (array_one[i] > threshold) and (array_two[i][j] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                    if folder == 2:
                        fp_sat_array[j].append(fp)
                        tn_sat_array[j].append(tn)
                    if folder == 1:
                        fn_fail_array[j].append(fn)
                        tp_fail_array[j].append(tp)
        else:
            for threshold in np.arange(0.0, 1.01, 0.01):
                fn = 0
                fp = 0
                tp = 0
                tn = 0
                for i in range(0, len(array_one), 1):
                    if folder == 2:
                        if (array_one[i] < threshold) and (array_two[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 1:
                        if (array_one[i] > threshold) and (array_two[i] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1
                if folder == 2:
                    fp_sat_array[0].append(fp)
                    tn_sat_array[0].append(tn)
                if folder == 1:
                    fn_fail_array[0].append(fn)
                    tp_fail_array[0].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        if type(array_one[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):
                for j in range(0, len(window_sizes), 1):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, len(array_one), 1):
                        if folder == 1:
                            if (array_one[i][j] < threshold) and (array_two[i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[i][j] > threshold) and (array_two[i] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[i][j] < threshold) and (array_two[i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[i][j] < threshold) and (array_two[i] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[j].append(fp)
                        tn_array_1[j].append(tn)
                    if folder == 2:
                        fn_array_2[j].append(fn)
                        tp_array_2[j].append(tp)
                    if folder == 3:
                        fp_array_3[j].append(fp)
                        tn_array_3[j].append(tn)
                    if folder == 4:
                        fp_array_4[j].append(fp)
                        tn_array_4[j].append(tn)
        elif type(array_two[0]) is list:
            for threshold in np.arange(0.0, 1.01, 0.01):
                for j in range(0, len(window_sizes), 1):
                    fn = 0
                    fp = 0
                    tp = 0
                    tn = 0
                    for i in range(0, len(array_one), 1):
                        if folder == 1:
                            if (array_one[i] < threshold) and (array_two[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 2:
                            if (array_one[i] > threshold) and (array_two[i][j] > threshold):
                                fn = fn + 1
                            else:
                                tp = tp + 1

                        elif folder == 3:
                            if (array_one[i] < threshold) and (array_two[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                        elif folder == 4:
                            if (array_one[i] < threshold) and (array_two[i][j] < threshold):
                                fp = fp + 1
                            else:
                                tn = tn + 1

                    if folder == 1:
                        fp_array_1[j].append(fp)
                        tn_array_1[j].append(tn)
                    if folder == 2:
                        fn_array_2[j].append(fn)
                        tp_array_2[j].append(tp)
                    if folder == 3:
                        fp_array_3[j].append(fp)
                        tn_array_3[j].append(tn)
                    if folder == 4:
                        fp_array_4[j].append(fp)
                        tn_array_4[j].append(tn)
        else:
            for threshold in np.arange(0.0, 1.01, 0.01):
                fn = 0
                fp = 0
                tp = 0
                tn = 0
                for i in range(0, len(array_one), 1):
                    if folder == 1:
                        if (array_one[i] < threshold) and (array_two[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 2:
                        if (array_one[i] > threshold) and (array_two[i] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1

                    elif folder == 3:
                        if (array_one[i] < threshold) and (array_two[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 4:
                        if (array_one[i] < threshold) and (array_two[i] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                if folder == 1:
                    fp_array_1[0].append(fp)
                    tn_array_1[0].append(tn)
                if folder == 2:
                    fn_array_2[0].append(fn)
                    tp_array_2[0].append(tp)
                if folder == 3:
                    fp_array_3[0].append(fp)
                    tn_array_3[0].append(tn)
                if folder == 4:
                    fp_array_4[0].append(fp)
                    tn_array_4[0].append(tn)

    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


def all_IM(array_one, array_two, array_three, selected_dataset, folder,splits):
    if selected_dataset == 1:
        for threshold in np.arange(0.0, 1.01, 0.01):
            # print "threshold = ", threshold
            for j in range(0, len(window_sizes), 1):
                fp = 0;
                tn = 0;
                fn = 0;
                tp = 0;
                for i in range(0, len(array_one), 1):
                    if folder == 2:
                        if (array_one[i] < threshold) and (array_two[i] < threshold) and (
                                array_three[i][j] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 1:
                        if (array_one[i] > threshold) and (array_two[i] > threshold) and (
                                array_three[i][j] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1

                if folder == 2:
                    fp_sat_array[j].append(fp)
                    tn_sat_array[j].append(tn)
                if folder == 1:
                    fn_fail_array[j].append(fn)
                    tp_fail_array[j].append(tp)

        return fp_sat_array, tn_sat_array, fn_fail_array, tp_fail_array

    elif selected_dataset == 2:
        for threshold in np.arange(0.0, 1.01, 0.01):
            for j in range(0, len(window_sizes), 1):
                fn = 0;
                fp = 0;
                tp = 0;
                tn = 0
                # print "threshold = ", threshold
                for i in range(0, (min([len(x) for x in array_one])), 1):
                    if folder == 1:
                        if (array_one[i] < threshold) and (array_two[i] < threshold) and (
                                array_three[j][i] < threshold):
                            print(j,i)
                            fp = fp + 1
                        else:
                            tn = tn + 1
                            print(j,  i)

                    elif folder == 2:
                        if (array_one[i] > threshold) and (array_two[i] > threshold) and (
                                array_three[i][j] > threshold):
                            fn = fn + 1
                        else:
                            tp = tp + 1

                    elif folder == 3:
                        if (array_one[i] < threshold) and (array_two[i] < threshold) and (
                                array_three[i][j] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                    elif folder == 4:
                        if (array_one[i] < threshold) and (array_two[i] < threshold) and (
                                array_three[i][j] < threshold):
                            fp = fp + 1
                        else:
                            tn = tn + 1

                if folder == 1:
                    fp_array_1[j].append(fp)
                    tn_array_1[j].append(tn)
                if folder == 2:
                    fn_array_2[j].append(fn)
                    tp_array_2[j].append(tp)
                if folder == 3:
                    fp_array_3[j].append(fp)
                    tn_array_3[j].append(tn)
                if folder == 4:
                    fp_array_4[j].append(fp)
                    tn_array_4[j].append(tn)



    return fp_array_1, tn_array_1, fn_array_2, tp_array_2, fp_array_3, tn_array_3, fp_array_4, tn_array_4


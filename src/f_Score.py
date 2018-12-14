from imageMatching import window_sizes
import numpy as np
import os

def f_score(array_1, array_2, array_3, array_4, array_5, array_6, array_7, array_8, selected_dataset, image_count,name,result_folder,temp_enable,splits):
    if temp_enable is True:
        length_of_windows = len(window_sizes)
    else:
        length_of_windows = 1

    fp_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    tn_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    threshold_array = []
    precision_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    recall_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    sensitivity_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    specificity_fn_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    accuracy_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    f1_score_calc_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    f2_score_calc_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    f05_score_calc_array = [[[0] * 101 for a in range(splits)]*1 for i in range(length_of_windows)]
    max_index_1_av = []
    max_index_2_av = []
    max_index_05_av = []
    max_index_1 = [[[0]*1 for i in range(splits)]*1 for i in range(length_of_windows)]
    max_index_2 = [[[0] * 1 for i in range(splits)] * 1 for i in range(length_of_windows)]
    max_index_05 = [[[0] * 1 for i in range(splits)] * 1 for i in range(length_of_windows)]
    if selected_dataset == 1:
        for k in range(0,splits,1):
            for j in range(0, length_of_windows, 1):
                threshold_array = []
                for i in range(0, 101, 1):
                    threshold_array.append(i)

                    try:
                        precision_array[k][j][i] = (float(array_4[k][j][i])) / float(array_4[k][j][i] + array_1[k][j][i])
                        recall_array[k][j][i] = float(array_4[k][j][i]) / float(array_4[k][j][i] + array_3[k][j][i])
                        specificity = (array_2[k][j][i]) / (array_2[k][j][i] + array_1[k][j][i])
                        specificity_fn_array[k][j][i] = 1 - specificity
                        sensitivity_array = recall_array
                    except ZeroDivisionError:
                        precision_array[k][j][i] = 0
                        sensitivity_array[k][j][i] = 0
                        recall_array[k][j][i] = 0
                        specificity_fn_array[k][j][i] = 1

                    accuracy_array[k][j][i] = float(array_2[k][j][i] + array_4[k][j][i]) / float(
                        array_2[k][j][i] + array_4[k][j][i] + array_1[k][j][i] + array_3[k][j][i])
                    try:
                        f1_score_calc_array[k][j][i] = 2 * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            precision_array[k][j][i] + recall_array[k][j][i])
                        f2_score_calc_array[k][j][i] = (1 + 4) * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            4 * precision_array[k][j][i] + recall_array[k][j][i])
                        f05_score_calc_array[k][j][i] = (1 + 0.25) * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            0.25 * precision_array[k][j][i] + recall_array[k][j][i])
                    except ZeroDivisionError:
                        f1_score_calc_array[k][j][i] = 0
                        f2_score_calc_array[k][j][i] = 0
                        f05_score_calc_array[k][j][i] = 0

                max_index_1[k][j] = f1_score_calc_array[k][j].index(max(f1_score_calc_array[k][j]))
                max_index_2[k][j] = f2_score_calc_array[k][j].index(max(f2_score_calc_array[k][j]))
                max_index_05[k][j] = f05_score_calc_array[k][j].index(max(f05_score_calc_array[k][j]))

                print "f1_score", max_index_1[k][j]
                print "f2_score", max_index_2[k][j]
                print "f0.5_score", max_index_05[k][j]

            max_index_1_av = np.mean(max_index_1[k])
            max_index_2_av = np.mean(max_index_2[k])
            max_index_05_av = np.mean(max_index_05[k])


        if not os.path.exists(result_folder + name + '.txt'):
            with open(result_folder + name + '.txt',
                      "w") as file:
                file.close()
        with open(result_folder + name + '.txt', "a") as file:
            if name in ('all', 'histogram_template', 'pixel_template', 'template'):
                for k in range(0, len(window_sizes), 1):
                    file.write("Window" + str(window_sizes[k]) + "\n")
                    file.write("fp_f1 = " + str(array_1[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tn_f1 = " + str(array_2[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f1 = " + str(array_3[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f1 = " + str(array_4[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_1[k]]) + "\n")
                    file.write("threshold = " + str(max_index_1[k]) + "\n")
                    file.write("\n")
                    file.write("fp_f2 = " + str(array_1[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f2 = " + str(array_2[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f2 = " + str(array_3[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f2 = " + str(array_4[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_2[k]]) + "\n")
                    file.write("threshold = " + str(max_index_2[k]) + "\n")
                    file.write("\n")
                    file.write("fp_f05 = " + str(array_1[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f05 = " + str(array_2[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f05 = " + str(array_3[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f05 = " + str(array_4[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_05[k]]) + "\n")
                    file.write("threshold = " + str(max_index_05) + "\n")
                    file.write("\n")
            else:
                for k in range(0, len(window_sizes), 1):
                    file.write("fp_f1 = " + str(array_1[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tn_f1 = " + str(array_2[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f1 = " + str(array_3[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f1 = " + str(array_4[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_1[k]]) + "\n")
                    file.write("threshold = " + str(max_index_1[k]) + "\n")
                    file.write("\n")
                    file.write("fp_f2 = " + str(array_1[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f2 = " + str(array_2[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f2 = " + str(array_3[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f2 = " + str(array_4[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_2[k]]) + "\n")
                    file.write("threshold = " + str(max_index_2[k]) + "\n")
                    file.write("\n")
                    file.write("fp_f05 = " + str(array_1[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f05 = " + str(array_2[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f05 = " + str(array_3[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f05 = " + str(array_4[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[k][max_index_05[k]]) + "\n")
                    file.write("threshold = " + str(max_index_05[k]) + "\n")
                    file.write("\n")
            file.close()
        return f1_score_calc_array, f2_score_calc_array, f05_score_calc_array, threshold_array, precision_array, recall_array, sensitivity_array, specificity_fn_array, accuracy_array

    elif selected_dataset == 2:
        for k in range(0, length_of_windows, 1):
            for j in range(0, splits, 1):
                threshold_array = []
                for i in range(0, 101, 1):
                    threshold_array.append(i)
                    fp_array[k][j][i] = array_1[0][k][j][i] + array_5[0][k][j][i] + array_7[0][k][j][i]
                    tn_array[k][j][i] = array_2[0][k][j][i] + array_6[0][k][j][i] + array_8[0][k][j][i]

                    try:
                        precision_array[k][j][i] = (float(array_4[0][k][j][i])) / float(array_4[0][k][j][i] + fp_array[k][j][i])
                        recall_array[k][j][i] = float(array_4[0][k][j][i]) / float(array_4[0][k][j][i] + array_3[0][k][j][i])
                        specificity = float(tn_array[k][j][i]) / float(tn_array[k][j][i] + fp_array[k][j][i])
                        specificity_fn_array[k][j][i] = 1.0 - specificity
                        sensitivity_array = recall_array
                    except ZeroDivisionError:
                        precision_array[k][j][i] = 0
                        sensitivity_array[k][j][i] = 0
                        recall_array[k][j][i] = 0
                        specificity_fn_array[k][j][i] = 1

                    accuracy_array[k][j][i] = float(tn_array[k][j][i] + array_4[0][k][j][i]) / float(
                        tn_array[k][j][i] + array_4[0][k][j][i] + fp_array[k][j][i] + array_3[0][k][j][i])
                    try:
                        f1_score_calc_array[k][j][i] = 2 * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            precision_array[k][j][i] + recall_array[k][j][i])
                        f2_score_calc_array[k][j][i] = (1 + 4) * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            4 * precision_array[k][j][i] + recall_array[k][j][i])
                        f05_score_calc_array[k][j][i] = (1 + 0.25) * float(precision_array[k][j][i] * recall_array[k][j][i]) / float(
                            0.25 * precision_array[k][j][i] + recall_array[k][j][i])
                    except ZeroDivisionError:
                        f1_score_calc_array[k][j][i] = 0
                        f2_score_calc_array[k][j][i] = 0
                        f05_score_calc_array[k][j][i] = 0

                max_index_1[k][j] = f1_score_calc_array[k][j].index(max(f1_score_calc_array[k][j]))
                max_index_2[k][j] = f2_score_calc_array[k][j].index(max(f2_score_calc_array[k][j]))
                max_index_05[k][j] = f05_score_calc_array[k][j].index(max(f05_score_calc_array[k][j]))

                #print "f1_score", max_index_1[k][j]
                #print "f2_score", max_index_2[k][j]
                #print "f0.5_score", max_index_05[k][j]


        for i in range (length_of_windows):
            temp = []
            for row in max_index_1:
                if row is not None:
                    temp.append(row[i])
            print temp
            mean = float(sum(temp)) / len(temp)
            max_index_1_av.append(mean)
            temp = []
            for row in max_index_2:
                temp.append(row[i])
            print temp
            mean = float(sum(temp)) / len(temp)
            max_index_2_av.append(mean)
            temp = []
            for row in max_index_05:
                temp.append(row[i])
            print temp
            mean = float(sum(temp)) / len(temp)
            max_index_05_av.append(mean)


        #if not os.path.exists(result_folder + name + '.txt'):
        #    with open(result_folder + name + '.txt', "w") as file:
        #        file.close()
        #with open(result_folder + name + '.txt', "a") as file:
        #    if name in ('all', 'histogram_template', 'pixel_template', 'template'):
        #        for k in range(0, len(window_sizes), 1):
        #            file.write("Window" + str(window_sizes[k]) + "\n")
        #            file.write("fp_f1 = " + str(array_1[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tn_f1 = " + str(array_2[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f1 = " + str(array_3[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f1 = " + str(array_4[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f1 = " + str(array_5[k][max_index_1[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tn_f1 = " + str(array_6[k][max_index_1[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f1 = " + str(array_7[k][max_index_1[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tn_f1 = " + str(array_8[k][max_index_1[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_1[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_1[k]) + "\n")
        #            file.write("\n")
        #            file.write("fp_f2 = " + str(array_1[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tp_f2 = " + str(array_2[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f2 = " + str(array_3[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f2 = " + str(array_4[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f2 = " + str(array_5[k][max_index_2[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tp_f2 = " + str(array_6[k][max_index_2[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f2 = " + str(array_7[k][max_index_2[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tp_f2 = " + str(array_8[k][max_index_2[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_2[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_2[k]) + "\n")
        #            file.write("\n")
        #            file.write("fp_f05 = " + str(array_1[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tp_f05 = " + str(array_2[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f05 = " + str(array_3[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f05 = " + str(array_4[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f05 = " + str(array_5[k][max_index_05[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tp_f05 = " + str(array_6[k][max_index_05[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f05 = " + str(array_7[k][max_index_05[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tp_f05 = " + str(array_8[k][max_index_05[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_05[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_05[k]) + "\n")
        #            file.write("\n")
        #    else:
        #        for k in range(0, len(window_sizes), 1):
        #            file.write("Window" + str(window_sizes[k]) + "\n")
        #            file.write("fp_f1 = " + str(array_1[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tn_f1 = " + str(array_2[k][max_index_1[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f1 = " + str(array_3[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f1 = " + str(array_4[k][max_index_1[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f1 = " + str(array_5[k][max_index_1[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tn_f1 = " + str(array_6[k][max_index_1[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f1 = " + str(array_7[k][max_index_1[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tn_f1 = " + str(array_8[k][max_index_1[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_1[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_1[j]) + "\n")
        #            file.write("\n")
        #            file.write("fp_f2 = " + str(array_1[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tp_f2 = " + str(array_2[k][max_index_2[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f2 = " + str(array_3[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f2 = " + str(array_4[k][max_index_2[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f2 = " + str(array_5[k][max_index_2[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tp_f2 = " + str(array_6[k][max_index_2[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f2 = " + str(array_7[k][max_index_2[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tp_f2 = " + str(array_8[k][max_index_2[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_2[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_2[j]) + "\n")
        #            file.write("\n")
        #            file.write("fp_f05 = " + str(array_1[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("tp_f05 = " + str(array_2[k][max_index_05[k]]) + '/' + str(image_count[0]) + "\n")
        #            file.write("fn_f05 = " + str(array_3[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("tp_f05 = " + str(array_4[k][max_index_05[k]]) + '/' + str(image_count[1]) + "\n")
        #            file.write("fp_f05 = " + str(array_5[k][max_index_05[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("tp_f05 = " + str(array_6[k][max_index_05[k]]) + '/' + str(image_count[2]) + "\n")
        #            file.write("fp_f05 = " + str(array_7[k][max_index_05[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("tp_f05 = " + str(array_8[k][max_index_05[k]]) + '/' + str(image_count[3]) + "\n")
        #            file.write("accuracy = " + str(accuracy_array[k][max_index_05[k]]) + "\n")
        #            file.write("threshold = " + str(max_index_05[j]) + "\n")
        #            file.write("\n")
        #    file.close()

        return f1_score_calc_array, f2_score_calc_array, f05_score_calc_array, threshold_array, precision_array, recall_array, sensitivity_array, specificity_fn_array, accuracy_array, max_index_1_av,max_index_2_av,max_index_05_av

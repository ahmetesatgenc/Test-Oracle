from imageMatching import window_sizes
import os




def f_score(array_1, array_2, array_3, array_4, array_5, array_6, array_7, array_8, selected_dataset, image_count,name,result_folder,temp_enable):
    fp_array = [[0] * 101 for a in range(len(window_sizes))]
    tn_array = [[0] * 101 for a in range(len(window_sizes))]
    threshold_array = []
    precision_array = [[0] * 101 for a in range(len(window_sizes))]
    recall_array = [[0] * 101 for a in range(len(window_sizes))]
    sensitivity_array = [[0] * 101 for a in range(len(window_sizes))]
    specificity_fn_array = [[0] * 101 for a in range(len(window_sizes))]
    accuracy_array = [[0] * 101 for a in range(len(window_sizes))]
    f1_score_calc_array = [[0] * 101 for a in range(len(window_sizes))]
    f2_score_calc_array = [[0] * 101 for a in range(len(window_sizes))]
    f05_score_calc_array = [[0] * 101 for a in range(len(window_sizes))]
    length_of_windows = len(window_sizes)
    if selected_dataset == 1:
        if temp_enable is False:
            length_of_windows = 1
        for j in range(0, length_of_windows, 1):
            threshold_array = []
            for i in range(0, 101, 1):
                threshold_array.append(i)

                try:
                    precision_array[j][i] = (float(array_4[j][i])) / float(array_4[j][i] + array_1[j][i])
                    recall_array[j][i] = float(array_4[j][i]) / float(array_4[j][i] + array_3[j][i])
                    specificity = (array_2[j][i]) / (array_2[j][i] + array_1[j][i])
                    specificity_fn_array[j][i] = 1 - specificity
                    sensitivity_array = recall_array
                except ZeroDivisionError:
                    precision_array[j][i] = 0
                    sensitivity_array[j][i] = 0
                    recall_array[j][i] = 0
                    specificity_fn_array[j][i] = 1

                accuracy_array[j][i] = float(array_2[j][i] + array_4[j][i]) / float(
                    array_2[j][i] + array_4[j][i] + array_1[j][i] + array_3[j][i])
                try:
                    f1_score_calc_array[j][i] = 2 * float(precision_array[j][i] * recall_array[j][i]) / float(
                        precision_array[j][i] + recall_array[j][i])
                    f2_score_calc_array[j][i] = (1 + 4) * float(precision_array[j][i] * recall_array[j][i]) / float(
                        4 * precision_array[j][i] + recall_array[j][i])
                    f05_score_calc_array[j][i] = (1 + 0.25) * float(precision_array[j][i] * recall_array[j][i]) / float(
                        0.25 * precision_array[j][i] + recall_array[j][i])
                except ZeroDivisionError:
                    f1_score_calc_array[j][i] = 0
                    f2_score_calc_array[j][i] = 0
                    f05_score_calc_array[j][i] = 0

        max_index_1 = f1_score_calc_array[j].index(max(f1_score_calc_array[j]))
        print "f1_score", max_index_1
        max_index_2 = f2_score_calc_array[j].index(max(f2_score_calc_array[j]))
        print "f2_score", max_index_2
        max_index_05 = f05_score_calc_array[j].index(max(f05_score_calc_array[j]))
        print "f05_score", max_index_05
        if not os.path.exists(result_folder + name + '.txt'):
            with open(result_folder + name + '.txt',
                      "w") as file:
                file.close()
        with open(result_folder + name + '.txt', "a") as file:
            if name in ('all', 'histogram_template', 'pixel_template', 'template'):
                for j in range(0, len(window_sizes), 1):
                    file.write("Window" + str(window_sizes[j]) + "\n")
                    file.write("fp_f1 = " + str(array_1[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                    file.write("tn_f1 = " + str(array_2[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f1 = " + str(array_3[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f1 = " + str(array_4[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_1]) + "\n")
                    file.write("threshold = " + str(max_index_1) + "\n")
                    file.write("\n")
                    file.write("fp_f2 = " + str(array_1[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f2 = " + str(array_2[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f2 = " + str(array_3[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f2 = " + str(array_4[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_2]) + "\n")
                    file.write("threshold = " + str(max_index_2) + "\n")
                    file.write("\n")
                    file.write("fp_f05 = " + str(array_1[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f05 = " + str(array_2[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                    file.write("fn_f05 = " + str(array_3[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f05 = " + str(array_4[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_05]) + "\n")
                    file.write("threshold = " + str(max_index_05) + "\n")
                    file.write("\n")
            else:
                file.write("fp_f1 = " + str(array_1[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                file.write("tn_f1 = " + str(array_2[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                file.write("fn_f1 = " + str(array_3[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                file.write("tp_f1 = " + str(array_4[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_1]) + "\n")
                file.write("threshold = " + str(max_index_1) + "\n")
                file.write("\n")
                file.write("fp_f2 = " + str(array_1[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                file.write("tp_f2 = " + str(array_2[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                file.write("fn_f2 = " + str(array_3[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                file.write("tp_f2 = " + str(array_4[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_2]) + "\n")
                file.write("threshold = " + str(max_index_2) + "\n")
                file.write("\n")
                file.write("fp_f05 = " + str(array_1[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                file.write("tp_f05 = " + str(array_2[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                file.write("fn_f05 = " + str(array_3[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                file.write("tp_f05 = " + str(array_4[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_05]) + "\n")
                file.write("threshold = " + str(max_index_05) + "\n")
                file.write("\n")
            file.close()
        return f1_score_calc_array, f2_score_calc_array, f05_score_calc_array, threshold_array, precision_array, recall_array, sensitivity_array, specificity_fn_array, accuracy_array

    elif selected_dataset == 2:
        if temp_enable is False:
            length_of_windows = 1
        for j in range(0, length_of_windows, 1):
            threshold_array = []
            for i in range(0, 101, 1):
                threshold_array.append(i)
                fp_array[j][i] = array_1[j][i] + array_5[j][i] + array_7[j][i]
                tn_array[j][i] = array_2[j][i] + array_6[j][i] + array_8[j][i]

                try:
                    precision_array[j][i] = (float(array_4[j][i])) / float(array_4[j][i] + fp_array[j][i])
                    recall_array[j][i] = float(array_4[j][i]) / float(array_4[j][i] + array_3[j][i])
                    specificity = float(tn_array[j][i]) / float(tn_array[j][i] + fp_array[j][i])
                    specificity_fn_array[j][i] = 1.0 - specificity
                    sensitivity_array = recall_array
                except ZeroDivisionError:
                    precision_array[j][i] = 0
                    sensitivity_array[j][i] = 0
                    recall_array[j][i] = 0
                    specificity_fn_array[j][i] = 1

                accuracy_array[j][i] = float(tn_array[j][i] + array_4[j][i]) / float(
                    tn_array[j][i] + array_4[j][i] + fp_array[j][i] + array_3[j][i])
                try:
                    f1_score_calc_array[j][i] = 2 * float(precision_array[j][i] * recall_array[j][i]) / float(
                        precision_array[j][i] + recall_array[j][i])
                    f2_score_calc_array[j][i] = (1 + 4) * float(precision_array[j][i] * recall_array[j][i]) / float(
                        4 * precision_array[j][i] + recall_array[j][i])
                    f05_score_calc_array[j][i] = (1 + 0.25) * float(precision_array[j][i] * recall_array[j][i]) / float(
                        0.25 * precision_array[j][i] + recall_array[j][i])
                except ZeroDivisionError:
                    f1_score_calc_array[j][i] = 0
                    f2_score_calc_array[j][i] = 0
                    f05_score_calc_array[j][i] = 0

        max_index_1 = f1_score_calc_array[j].index(max(f1_score_calc_array[j]))
        print "f1_score", max_index_1
        max_index_2 = f2_score_calc_array[j].index(max(f2_score_calc_array[j]))
        print "f2_score", max_index_2
        max_index_05 = f05_score_calc_array[j].index(max(f05_score_calc_array[j]))
        print "f05_score", max_index_05
        if not os.path.exists(result_folder + name + '.txt'):
            with open(result_folder + name + '.txt', "w") as file:
                file.close()
        with open(result_folder + name + '.txt', "a") as file:
            if name in ('all', 'histogram_template', 'pixel_template', 'template'):
                for j in range(0, len(window_sizes), 1):
                    file.write("Window" + str(window_sizes[j]) + "\n")
                    file.write("fp_f1 = " + str(array_1[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                    file.write("tn_f1 = " + str(array_2[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                    file.write("fn_f1 = " + str(array_3[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f1 = " + str(array_4[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                    file.write("fp_f1 = " + str(array_5[j][max_index_1]) + '/' + str(image_count[2]) + "\n")
                    file.write("tn_f1 = " + str(array_6[j][max_index_1]) + '/' + str(image_count[2]) + "\n")
                    file.write("fp_f1 = " + str(array_7[j][max_index_1]) + '/' + str(image_count[3]) + "\n")
                    file.write("tn_f1 = " + str(array_8[j][max_index_1]) + '/' + str(image_count[3]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_1]) + "\n")
                    file.write("threshold = " + str(max_index_1) + "\n")
                    file.write("\n")
                    file.write("fp_f2 = " + str(array_1[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f2 = " + str(array_2[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                    file.write("fn_f2 = " + str(array_3[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f2 = " + str(array_4[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                    file.write("fp_f2 = " + str(array_5[j][max_index_2]) + '/' + str(image_count[2]) + "\n")
                    file.write("tp_f2 = " + str(array_6[j][max_index_2]) + '/' + str(image_count[2]) + "\n")
                    file.write("fp_f2 = " + str(array_7[j][max_index_2]) + '/' + str(image_count[3]) + "\n")
                    file.write("tp_f2 = " + str(array_8[j][max_index_2]) + '/' + str(image_count[3]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_2]) + "\n")
                    file.write("threshold = " + str(max_index_2) + "\n")
                    file.write("\n")
                    file.write("fp_f05 = " + str(array_1[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                    file.write("tp_f05 = " + str(array_2[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                    file.write("fn_f05 = " + str(array_3[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                    file.write("tp_f05 = " + str(array_4[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                    file.write("fp_f05 = " + str(array_5[j][max_index_05]) + '/' + str(image_count[2]) + "\n")
                    file.write("tp_f05 = " + str(array_6[j][max_index_05]) + '/' + str(image_count[2]) + "\n")
                    file.write("fp_f05 = " + str(array_7[j][max_index_05]) + '/' + str(image_count[3]) + "\n")
                    file.write("tp_f05 = " + str(array_8[j][max_index_05]) + '/' + str(image_count[3]) + "\n")
                    file.write("accuracy = " + str(accuracy_array[j][max_index_05]) + "\n")
                    file.write("threshold = " + str(max_index_05) + "\n")
                    file.write("\n")
            else:
                file.write("fp_f1 = " + str(array_1[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                file.write("tn_f1 = " + str(array_2[j][max_index_1]) + '/' + str(image_count[0]) + "\n")
                file.write("fn_f1 = " + str(array_3[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                file.write("tp_f1 = " + str(array_4[j][max_index_1]) + '/' + str(image_count[1]) + "\n")
                file.write("fp_f1 = " + str(array_5[j][max_index_1]) + '/' + str(image_count[2]) + "\n")
                file.write("tn_f1 = " + str(array_6[j][max_index_1]) + '/' + str(image_count[2]) + "\n")
                file.write("fp_f1 = " + str(array_7[j][max_index_1]) + '/' + str(image_count[3]) + "\n")
                file.write("tn_f1 = " + str(array_8[j][max_index_1]) + '/' + str(image_count[3]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_1]) + "\n")
                file.write("threshold = " + str(max_index_1) + "\n")
                file.write("\n")
                file.write("fp_f2 = " + str(array_1[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                file.write("tp_f2 = " + str(array_2[j][max_index_2]) + '/' + str(image_count[0]) + "\n")
                file.write("fn_f2 = " + str(array_3[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                file.write("tp_f2 = " + str(array_4[j][max_index_2]) + '/' + str(image_count[1]) + "\n")
                file.write("fp_f2 = " + str(array_5[j][max_index_2]) + '/' + str(image_count[2]) + "\n")
                file.write("tp_f2 = " + str(array_6[j][max_index_2]) + '/' + str(image_count[2]) + "\n")
                file.write("fp_f2 = " + str(array_7[j][max_index_2]) + '/' + str(image_count[3]) + "\n")
                file.write("tp_f2 = " + str(array_8[j][max_index_2]) + '/' + str(image_count[3]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_2]) + "\n")
                file.write("threshold = " + str(max_index_2) + "\n")
                file.write("\n")
                file.write("fp_f05 = " + str(array_1[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                file.write("tp_f05 = " + str(array_2[j][max_index_05]) + '/' + str(image_count[0]) + "\n")
                file.write("fn_f05 = " + str(array_3[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                file.write("tp_f05 = " + str(array_4[j][max_index_05]) + '/' + str(image_count[1]) + "\n")
                file.write("fp_f05 = " + str(array_5[j][max_index_05]) + '/' + str(image_count[2]) + "\n")
                file.write("tp_f05 = " + str(array_6[j][max_index_05]) + '/' + str(image_count[2]) + "\n")
                file.write("fp_f05 = " + str(array_7[j][max_index_05]) + '/' + str(image_count[3]) + "\n")
                file.write("tp_f05 = " + str(array_8[j][max_index_05]) + '/' + str(image_count[3]) + "\n")
                file.write("accuracy = " + str(accuracy_array[j][max_index_05]) + "\n")
                file.write("threshold = " + str(max_index_05) + "\n")
                file.write("\n")
            file.close()

        return f1_score_calc_array, f2_score_calc_array, f05_score_calc_array, threshold_array, precision_array, recall_array, sensitivity_array, specificity_fn_array, accuracy_array

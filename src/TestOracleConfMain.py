import os
import glob
from featureDetector import *
from descriptorExtractor import *
from transformation import *
from imageMatching import *
from matcher import BFmatcher
from binaryClassification import *
from f_Score import f_score
from plotGraph import plot_Graph
from boxPlot import box_plot
from readConfig import get_Feature


np.set_printoptions(threshold='nan')

def load_images_from_folder(folder):
    grab_name = "grab.png"
    ref_name = "ref.png"
    for filename in os.listdir(folder):
        if filename == grab_name:
            print "file name:", os.path.join(folder, filename)
            img_grab = cv2.imread(os.path.join(folder, filename))
        elif filename == ref_name:
            print "file name:", os.path.join(folder, filename)
            img_ref = cv2.imread(os.path.join(folder, filename))
    return img_grab, img_ref

def datasetSelect(dataset):
    if dataset == 1:
        main_folder = './sample_images/dataset_1/'
        folder_list = os.listdir(main_folder)
        folder_dict = {folder_list[0]: 5, folder_list[1]: 5}
        selected_dataset = 1
    elif dataset == 2:
        main_folder = './sample_images/dataset_2/'
        folder_list = os.listdir(main_folder)
        folder_dict = {folder_list[0]: 5, folder_list[1]: 5, folder_list[2]: 5,
                       folder_list[3]: 5}
        selected_dataset = 2
    return main_folder, folder_dict, selected_dataset


def main():

    main_folder, folder_dict, selected_dataset = datasetSelect(dataset=1)
    a = 0
    hist_enable = False
    temp_enable= False
    pix_enable = False
    feature_detector, descriptor_extractor, transformation, image_matching = get_Feature()
    print "feature detector is: ", feature_detector
    print "descriptor extractor is:", descriptor_extractor
    print "transformation function is:", transformation
    print "image matching algorithm is:", image_matching
    feature_mapping = {'FAST':FAST_feature,'STAR':STAR_feature,'SIFT':SIFT_feature,'SURF':SURF_feature,'ORB':ORB_feature,
                       'BRISK':BRISK_feature,'SimpleBlob':SIMPLEBLOB_feature}
    descriptor_mapping = {'SIFT':SIFT_desc_extract,'SURF':SURF_desc_extract,'BRIEF':BRIEF_desc_extract,'BRISK':BRISK_desc_extract,
                          'ORB':ORB_desc_extract,'FREAK':FREAK_desc_extract}
    transformation_mapping = {'Affine Transform':Affine_transform, 'Perspective Transform':Perspective_transform}
    image_matching_mapping = {'Histogram Matching':Histogram_matching,'Template Matching':Template_Matching,'Pixel Matching':Pixel_matching}

    if len(image_matching) is 1:
        if image_matching[0] == 'Histogram Matching':
            hist_enable = True
            image_matching_1 = image_matching_mapping[image_matching[0]]
        elif image_matching[0] == 'Template Matching':
            temp_enable = True
            image_matching_2 = image_matching_mapping[image_matching[0]]
        elif image_matching[0] == 'Pixel Matching':
            pix_enable = True
            image_matching_3 = image_matching_mapping[image_matching[0]]
    elif len(image_matching) is 2:
        for k in range(0,2):
            if image_matching[k] == 'Histogram Matching':
                hist_enable = True
                image_matching_1 = image_matching_mapping[image_matching[k]]
            elif image_matching[k] == 'Template Matching':
                temp_enable = True
                image_matching_2 = image_matching_mapping[image_matching[k]]
            elif image_matching[k] == 'Pixel Matching':
                pix_enable = True
                image_matching_3 = image_matching_mapping[image_matching[k]]

    elif len(image_matching) is 3:
        hist_enable = True
        temp_enable = True
        pix_enable = True
        image_matching_1 = image_matching_mapping[image_matching[0]]
        image_matching_2 = image_matching_mapping[image_matching[1]]
        image_matching_3 = image_matching_mapping[image_matching[2]]

    if len(image_matching) is 3:
        result_folder = './test_oracle_result/'+ 'dataset_'+str(selected_dataset)+'/'+feature_detector +'_'+ descriptor_extractor +'_'+ transformation +'_'+\
                        image_matching[0]+'_'+image_matching[1]+'_'+image_matching[2]+'/'
    elif len(image_matching) is 2:
        result_folder = './test_oracle_result/'+'dataset_'+str(selected_dataset)+'/'+ feature_detector +'_'+ descriptor_extractor + '_'+transformation + '_'+\
                    image_matching[0]+'_'+ image_matching[1]+'/'
    else:
        result_folder = './test_oracle_result/' + 'dataset_'+str(selected_dataset)+'/'+ feature_detector + '_'+descriptor_extractor +'_'+ transformation +'_'+ \
                    image_matching[0]+'/'

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    else:
        files = glob.glob(result_folder+'/*')
        for f in files:
            os.remove(f)

    feature_detector = feature_mapping[feature_detector]
    descriptor_extractor = descriptor_mapping[descriptor_extractor]
    transformation = transformation_mapping[transformation]
    # with open(main_folder + "result/" + "result" + ".txt", "w") as file:
    #    file.close()

    for folder in range(1, len(folder_dict.values()) + 1, 1):
        hist_matching_array = []
        pixel_comp_array = []
        template_matching_array = []
        folder_count_array = []
        kp_grab_array = []
        kp_ref_array = []
        goodMatches_array = []
        image_count = folder_dict.values()
        folder_name = folder_dict.keys()
        for image in range(1, image_count[a] + 1, 1):

            folder_count_array.append(image)

            img_grab, img_ref = load_images_from_folder(main_folder + folder_name[a] + '/' + str(image) + '/')
            #kp_grab, des_grab = featureDetector(img_grab)
            #kp_ref, des_ref = featureDetector(img_ref)
            kp_grab = feature_detector(img_grab)
            kp_grab, des_grab = descriptor_extractor(img_grab,kp_grab)
            kp_grab_array.append(len(kp_grab))
            kp_ref = feature_detector(img_ref)
            kp_ref, des_ref = descriptor_extractor(img_ref,kp_ref)
            kp_ref_array.append(len(kp_ref))

            if kp_ref is not None and kp_grab is not None and des_grab is not None and des_ref is not None:

                matches, goodMatches = BFmatcher(des_grab, des_ref)
                goodMatches_array.append(len(goodMatches))
                if len(goodMatches):
                    warpedImage = transformation(kp_grab, kp_ref, img_grab, goodMatches)
            else:
                goodMatches_array.append(0)

            if hist_enable is True:
                hist_result = image_matching_1(img_ref, warpedImage)
                hist_matching_array.append(hist_result)
            if pix_enable is True:
                pixel_comp = image_matching_3(img_ref, warpedImage)
                pixel_comp_array.append(float(pixel_comp))
            if temp_enable is True:
                temp_result = image_matching_2(img_ref, warpedImage)
                template_matching_array.append(temp_result)

        a += 1
        if pix_enable is True:
            pixel_comp_array /= np.max(np.abs(pixel_comp_array), axis=0)
            pixel_comp_array = 1 - pixel_comp_array
            #pixel_comp_array = [(1 - float(i) / max(pixel_comp_array)) for i in pixel_comp_array]
            pixel_comp_array = pixel_comp_array.tolist()
            print pixel_comp_array
        if selected_dataset == 1:
            if hist_enable and temp_enable and pix_enable is True:
                fp_sat_array_all, tn_sat_array_all, fn_fail_array_all, tp_fail_array_all = all_IM(
                    hist_matching_array, pixel_comp_array, template_matching_array, selected_dataset, folder)
            elif (hist_enable is True and pix_enable is True) and temp_enable is False:
                fp_sat_array_hp, tn_sat_array_hp, fn_fail_array_hp, tp_fail_array_hp = two_IM(hist_matching_array, pixel_comp_array, selected_dataset, folder)
            elif (hist_enable is True and temp_enable is True) and pix_enable is False:
               fp_sat_array_ht, tn_sat_array_ht, fn_fail_array_ht, tp_fail_array_ht = two_IM(hist_matching_array, template_matching_array, selected_dataset, folder)
            elif (pix_enable is True and temp_enable is True) and hist_enable is False:
                fp_sat_array_tp, tn_sat_array_tp, fn_fail_array_tp, tp_fail_array_tp = two_IM(template_matching_array, pixel_comp_array, selected_dataset, folder)
            elif hist_enable is True and (pix_enable is False and temp_enable is False):
                fp_sat_array_h, tn_sat_array_h, fn_fail_array_h, tp_fail_array_h = one_IM(hist_matching_array,selected_dataset, folder)
            elif temp_enable is True and (pix_enable is False and temp_enable is False):
                fp_sat_array_t, tn_sat_array_t, fn_fail_array_t, tp_fail_array_t = one_IM(template_matching_array, selected_dataset, folder)
            elif pix_enable is True and (hist_enable is False and temp_enable is False):
                fp_sat_array_p, tn_sat_array_p, fn_fail_array_p, tp_fail_array_p = one_IM(pixel_comp_array, selected_dataset, folder)

        elif selected_dataset == 2:
            if hist_enable and temp_enable and pix_enable is True:
                fp_array_1_all, tn_array_1_all, fn_array_2_all, tp_array_2_all, fp_array_3_all, tn_array_3_all, fp_array_4_all, tn_array_4_all = all_IM(
                    hist_matching_array, pixel_comp_array, template_matching_array, selected_dataset, folder)
            elif (hist_enable is True and pix_enable is True) and temp_enable is False:
                fp_array_1_hp, tn_array_1_hp, fn_array_2_hp, tp_array_2_hp, fp_array_3_hp, tn_array_3_hp, fp_array_4_hp, tn_array_4_hp = two_IM(
                    hist_matching_array, pixel_comp_array,selected_dataset, folder)
            elif (hist_enable is True and temp_enable is True) and pix_enable is False:
                fp_array_1_ht, tn_array_1_ht, fn_array_2_ht, tp_array_2_ht, fp_array_3_ht, tn_array_3_ht, fp_array_4_ht, tn_array_4_ht = two_IM(
                    hist_matching_array, template_matching_array, selected_dataset, folder)
            elif (pix_enable is True and temp_enable is True) and hist_enable is False:
                fp_array_1_tp, tn_array_1_tp, fn_array_2_tp, tp_array_2_tp, fp_array_3_tp, tn_array_3_tp, fp_array_4_tp, tn_array_4_tp = two_IM(
                    template_matching_array, pixel_comp_array, selected_dataset, folder)
            elif hist_enable is True and (temp_enable is False and pix_enable is False):
                fp_array_1_h, tn_array_1_h, fn_array_2_h, tp_array_2_h, fp_array_3_h, tn_array_3_h, fp_array_4_h, tn_array_4_h = one_IM(
                    hist_matching_array, selected_dataset, folder)
            elif temp_enable is True and (hist_enable is False and pix_enable is False):
                fp_array_1_t, tn_array_1_t, fn_array_2_t, tp_array_2_t, fp_array_3_t, tn_array_3_t, fp_array_4_t, tn_array_4_t = one_IM(
                    template_matching_array, selected_dataset, folder)
            elif pix_enable is True and (temp_enable is False and hist_enable is False):
                fp_array_1_p, tn_array_1_p, fn_array_2_p, tp_array_2_p, fp_array_3_p, tn_array_3_p, fp_array_4_p, tn_array_4_p = one_IM(
                    pixel_comp_array, selected_dataset, folder)

        if hist_enable is True:
            plot_Graph(hist_matching_array, folder_count_array, 1.1, max(folder_count_array) + 1,
                  result_folder + 'Histogram_Matching_' + folder_name[folder - 1]+'.png',
                  'Histogram Accuracy', 'Images')
        if temp_enable is True:
            for k in range(0, len(window_sizes), 1):
                plot_Graph(list(map(list, zip(*template_matching_array)))[k], folder_count_array, 1.1,
                      max(folder_count_array) + 1, result_folder + 'Template_Matching' +'_window' + str(
                          window_sizes[k]) + '-' + folder_name[folder - 1] +
                      '.png', 'Template Accuracy', 'Images')
        if pix_enable is True :
            plot_Graph(pixel_comp_array, folder_count_array, 1.1, max(folder_count_array) + 1, result_folder + 'Pixel_Matching_' + folder_name[folder - 1]+ '.png',
                  'Pixel Accuracy', 'Images')

        plot_Graph(kp_grab_array, folder_count_array, max(kp_grab_array)+100, max(folder_count_array) + 1, result_folder + 'Grabbed Key Points_' + folder_name[folder - 1]+ '.png',
                'Grabbed Image Key Points', 'Images')

        plot_Graph(kp_ref_array, folder_count_array, max(kp_ref_array)+100, max(folder_count_array) + 1, result_folder + 'Ref Key Points_' + folder_name[folder - 1]+ '.png',
                'Ref Image Key Points', 'Images')

        plot_Graph(goodMatches_array, folder_count_array, max(goodMatches_array)+100, max(folder_count_array) + 1, result_folder + 'GoodMatches_' + folder_name[folder - 1]+ '.png',
                'Good Matches', 'Images')

        plot_Graph(kp_grab_array, goodMatches_array, max(kp_grab_array)+100, max(goodMatches_array) + 100, result_folder + 'kp_vs_GoodMatches_' + folder_name[folder - 1]+ '.png',
                'Grabbed Key Points', 'Good Matches')


        if hist_enable and temp_enable and pix_enable is True:
            box_plot(hist_matching_array, pixel_comp_array, template_matching_array, result_folder + 'BoxPlot_All' +
                 '-' + folder_name[folder - 1] +'.png')
        elif (hist_enable is True and temp_enable is True) and pix_enable is False:
            box_plot(hist_matching_array, None, template_matching_array, result_folder + 'BoxPlot_ht' +
                 '-' + folder_name[folder - 1] +'.png')
        elif (hist_enable is True and pix_enable is True) and temp_enable is False:
            box_plot(hist_matching_array, pixel_comp_array, None, result_folder + 'BoxPlot_hp' +
                 '-' + folder_name[folder - 1] +'.png')
        elif (pix_enable is True and temp_enable is True) and hist_enable is False:
            box_plot(None, pixel_comp_array, template_matching_array, result_folder + 'BoxPlot_pt' +
                 '-' + folder_name[folder - 1] +'.png')
        elif hist_enable is True and (temp_enable is False and pix_enable is False):
            box_plot(hist_matching_array, None, None, result_folder + 'BoxPlot_h' +
                 '-' + folder_name[folder - 1] +'.png')
        elif temp_enable is True and (pix_enable is False and hist_enable is False):
            box_plot(None, None, template_matching_array, result_folder + 'BoxPlot_t' +
                 '-' + folder_name[folder - 1] +'.png')
        elif pix_enable is True and (hist_enable is False and temp_enable is False):
            box_plot(None, pixel_comp_array, None, result_folder + 'BoxPlot_p' +
                 '-' + folder_name[folder - 1] +'.png')

    if selected_dataset == 1:
        if hist_enable and temp_enable and pix_enable is True:
            f1_score_calc_array_all, f2_score_calc_array_all, f05_score_calc_array_all, threshold_array_all, precision_array_all, recall_array_all, sensitivity_array_all, specificity_fn_array_all, accuracy_array_all = f_score(
                fp_sat_array_all, tn_sat_array_all, fn_fail_array_all, tp_fail_array_all, None, None, None, None,
                selected_dataset, image_count, 'all',result_folder,temp_enable)
        elif (hist_enable is True and pix_enable is True) and temp_enable is False:
            f1_score_calc_array_hp, f2_score_calc_array_hp, f05_score_calc_array_hp, threshold_array_hp, precision_array_hp, recall_array_hp, sensitivity_array_hp, specificity_fn_array_hp, accuracy_array_hp = f_score(
                fp_sat_array_hp, tn_sat_array_hp, fn_fail_array_hp, tp_fail_array_hp, None, None, None, None,
                selected_dataset, image_count, 'histogram_pixel',result_folder,temp_enable)
        elif (hist_enable is True and temp_enable is True) and pix_enable is False:
            f1_score_calc_array_ht, f2_score_calc_array_ht, f05_score_calc_array_ht, threshold_array_ht, precision_array_ht, recall_array_ht, sensitivity_array_ht, specificity_fn_array_ht, accuracy_array_ht = f_score(
                fp_sat_array_ht, tn_sat_array_ht, fn_fail_array_ht, tp_fail_array_ht, None, None, None, None,
                selected_dataset, image_count, 'histogram_template',result_folder,temp_enable)
        elif (temp_enable is True and pix_enable is True) and hist_enable is False:
            f1_score_calc_array_tp, f2_score_calc_array_tp, f05_score_calc_array_tp, threshold_array_tp, precision_array_tp, recall_array_tp, sensitivity_array_tp, specificity_fn_array_tp, accuracy_array_tp = f_score(
                fp_sat_array_tp, tn_sat_array_tp, fn_fail_array_tp, tp_fail_array_tp, None, None, None, None,
                selected_dataset, image_count, 'pixel_template',result_folder,temp_enable)
        elif hist_enable is True and (pix_enable is False and temp_enable is False):
            f1_score_calc_array_h, f2_score_calc_array_h, f05_score_calc_array_h, threshold_array_h, precision_array_h, recall_array_h, sensitivity_array_h, specificity_fn_array_h, accuracy_array_h = f_score(
                fp_sat_array_h, tn_sat_array_h, fn_fail_array_h, tp_fail_array_h, None, None, None, None, selected_dataset,
                image_count, 'histogram',result_folder,temp_enable)
        elif temp_enable is True and (pix_enable is False and hist_enable is False):
            f1_score_calc_array_t, f2_score_calc_array_t, f05_score_calc_array_t, threshold_array_t, precision_array_t, recall_array_t, sensitivity_array_t, specificity_fn_array_t, accuracy_array_t = f_score(
                fp_sat_array_t, tn_sat_array_t, fn_fail_array_t, tp_fail_array_t, None, None, None, None, selected_dataset,
                image_count, 'template',result_folder,temp_enable)
        elif pix_enable is True and (temp_enable is False and hist_enable is False):
            f1_score_calc_array_p, f2_score_calc_array_p, f05_score_calc_array_p, threshold_array_p, precision_array_p, recall_array_p, sensitivity_array_p, specificity_fn_array_p, accuracy_array_p = f_score(
                fp_sat_array_p, tn_sat_array_p, fn_fail_array_p, tp_fail_array_p, None, None, None, None, selected_dataset,
                image_count, 'pixel',result_folder,temp_enable)
    elif selected_dataset == 2:
        if hist_enable and temp_enable and pix_enable is True:
            f1_score_calc_array_all, f2_score_calc_array_all, f05_score_calc_array_all, threshold_array_all, precision_array_all, recall_array_all, sensitivity_array_all, specificity_fn_array_all, accuracy_array_all = f_score(
                fp_array_1_all, tn_array_1_all, fn_array_2_all, tp_array_2_all, fp_array_3_all, tn_array_3_all,
                fp_array_4_all, tn_array_4_all, selected_dataset, image_count, 'all',result_folder,temp_enable)
        elif (hist_enable is True and pix_enable is True) and temp_enable is False:
            f1_score_calc_array_hp, f2_score_calc_array_hp, f05_score_calc_array_hp, threshold_array_hp, precision_array_hp, recall_array_hp, sensitivity_array_hp, specificity_fn_array_hp, accuracy_array_hp = f_score(
                fp_array_1_hp, tn_array_1_hp, fn_array_2_hp, tp_array_2_hp, fp_array_3_hp, tn_array_3_hp, fp_array_4_hp,
                tn_array_4_hp, selected_dataset, image_count, 'histogram_pixel',result_folder,temp_enable)
        elif (hist_enable is True and temp_enable is True) and pix_enable is False:
            f1_score_calc_array_ht, f2_score_calc_array_ht, f05_score_calc_array_ht, threshold_array_ht, precision_array_ht, recall_array_ht, sensitivity_array_ht, specificity_fn_array_ht, accuracy_array_ht = f_score(
                fp_array_1_ht, tn_array_1_ht, fn_array_2_ht, tp_array_2_ht, fp_array_3_ht, tn_array_3_ht, fp_array_4_ht,
                tn_array_4_ht, selected_dataset, image_count, 'histogram_template',result_folder,temp_enable)
        elif (temp_enable is True and pix_enable is True) and hist_enable is False:
            f1_score_calc_array_tp, f2_score_calc_array_tp, f05_score_calc_array_tp, threshold_array_tp, precision_array_tp, recall_array_tp, sensitivity_array_tp, specificity_fn_array_tp, accuracy_array_tp = f_score(
                fp_array_1_tp, tn_array_1_tp, fn_array_2_tp, tp_array_2_tp, fp_array_3_tp, tn_array_3_tp, fp_array_4_tp,
                tn_array_4_tp, selected_dataset, image_count, 'pixel_template',result_folder,temp_enable)
        elif hist_enable is True and (temp_enable is False and pix_enable is False):
            f1_score_calc_array_h, f2_score_calc_array_h, f05_score_calc_array_h, threshold_array_h, precision_array_h, recall_array_h, sensitivity_array_h, specificity_fn_array_h, accuracy_array_h = f_score(
                fp_array_1_h, tn_array_1_h, fn_array_2_h, tp_array_2_h, fp_array_3_h, tn_array_3_h, fp_array_4_h,
                tn_array_4_h, selected_dataset, image_count, 'histogram',result_folder,temp_enable)
        elif temp_enable is True and (pix_enable is False and hist_enable is False):
            f1_score_calc_array_t, f2_score_calc_array_t, f05_score_calc_array_t, threshold_array_t, precision_array_t, recall_array_t, sensitivity_array_t, specificity_fn_array_t, accuracy_array_t = f_score(
                fp_array_1_t, tn_array_1_t, fn_array_2_t, tp_array_2_t, fp_array_3_t, tn_array_3_t, fp_array_4_t,
                tn_array_4_t, selected_dataset, image_count, 'template',result_folder,temp_enable)
        elif pix_enable is True and (hist_enable is False and temp_enable is False):
            f1_score_calc_array_p, f2_score_calc_array_p, f05_score_calc_array_p, threshold_array_p, precision_array_p, recall_array_p, sensitivity_array_p, specificity_fn_array_p, accuracy_array_p = f_score(
                fp_array_1_p, tn_array_1_p, fn_array_2_p, tp_array_2_p, fp_array_3_p, tn_array_3_p, fp_array_4_p,
                tn_array_4_p, selected_dataset, image_count, 'pixel',result_folder,temp_enable)

    for i in range(0, len(window_sizes), 1):
        if hist_enable and temp_enable and pix_enable is True:
            plot_Graph(f1_score_calc_array_all[i], threshold_array_all, max(f1_score_calc_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'f1_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f1_score', 'threshold')
            plot_Graph(f2_score_calc_array_all[i], threshold_array_all, max(f2_score_calc_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'f2_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f2_score', 'threshold')
            plot_Graph(f05_score_calc_array_all[i], threshold_array_all, max(f05_score_calc_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'f05_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f05_score', 'threshold')
            plot_Graph(precision_array_all[i], recall_array_all[i], max(precision_array_all[i]) + 0.1,
                  max(recall_array_all[0]) + 0.1,
                  result_folder + 'precision_vs_recall_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'recall')
            plot_Graph(sensitivity_array_all[i], specificity_fn_array_all[i], max(sensitivity_array_all[i]) + 0.1,
                  max(specificity_fn_array_all[i]) + 0.1,
                  result_folder + 'sensitivity_vs_specificity_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'specificity')
            plot_Graph(sensitivity_array_all[i], threshold_array_all, max(sensitivity_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'sensitivity_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'threshold')
            plot_Graph(precision_array_all[i], threshold_array_all, max(precision_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'precision_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'threshold')
            plot_Graph(accuracy_array_all[i], threshold_array_all, max(accuracy_array_all[i]) + 0.1,
                  max(threshold_array_all) + 1,
                  result_folder + 'accuracy_vs_threshold_all' + '_window_' + str(
                      window_sizes[i]) + '.png', 'accuracy', 'threshold')
        elif (hist_enable is True and temp_enable is True) and pix_enable is False:
            plot_Graph(f1_score_calc_array_ht[i], threshold_array_ht, max(f1_score_calc_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'f1_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f1_score', 'threshold')
            plot_Graph(f2_score_calc_array_ht[i], threshold_array_ht, max(f2_score_calc_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'f2_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f2_score', 'threshold')
            plot_Graph(f05_score_calc_array_ht[i], threshold_array_ht, max(f05_score_calc_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'f05_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f05_score', 'threshold')
            plot_Graph(precision_array_ht[i], recall_array_ht[i], max(precision_array_ht[i]) + 0.1,
                  max(recall_array_ht[0]) + 0.1,
                  result_folder + 'precision_vs_recall_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'recall')
            plot_Graph(sensitivity_array_ht[i], specificity_fn_array_ht[i], max(sensitivity_array_ht[i]) + 0.1,
                  max(specificity_fn_array_ht[i]) + 0.1,
                  result_folder + 'sensitivity_vs_specificity_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'specificity')
            plot_Graph(sensitivity_array_ht[i], threshold_array_ht, max(sensitivity_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'sensitivity_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'threshold')
            plot_Graph(precision_array_ht[i], threshold_array_ht, max(precision_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'precision_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'threshold')
            plot_Graph(accuracy_array_ht[i], threshold_array_ht, max(accuracy_array_ht[i]) + 0.1,
                  max(threshold_array_ht) + 1,
                  result_folder + 'accuracy_vs_threshold_ht' + '_window_' + str(
                      window_sizes[i]) + '.png', 'accuracy', 'threshold')
        elif (temp_enable is True and pix_enable is True) and hist_enable is False:
            plot_Graph(f1_score_calc_array_tp[i], threshold_array_tp, max(f1_score_calc_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'f1_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f1_score', 'threshold')
            plot_Graph(f2_score_calc_array_tp[i], threshold_array_tp, max(f2_score_calc_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'f2_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f2_score', 'threshold')
            plot_Graph(f05_score_calc_array_tp[i], threshold_array_tp, max(f05_score_calc_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'f05_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f05_score', 'threshold')
            plot_Graph(precision_array_tp[i], recall_array_tp[i], max(precision_array_tp[i]) + 0.1,
                  max(recall_array_tp[0]) + 0.1,
                  result_folder + 'precision_vs_recall_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'recall')
            plot_Graph(sensitivity_array_tp[i], specificity_fn_array_tp[i], max(sensitivity_array_tp[i]) + 0.1,
                  max(specificity_fn_array_tp[i]) + 0.1,
                  result_folder + 'sensitivity_vs_specificity_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'specificity')
            plot_Graph(sensitivity_array_tp[i], threshold_array_tp, max(sensitivity_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'sensitivity_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'threshold')
            plot_Graph(precision_array_tp[i], threshold_array_tp, max(precision_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'precision_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'threshold')
            plot_Graph(accuracy_array_tp[i], threshold_array_tp, max(accuracy_array_tp[i]) + 0.1,
                  max(threshold_array_tp) + 1,
                  result_folder + 'accuracy_vs_threshold_tp' + '_window_' + str(
                      window_sizes[i]) + '.png', 'accuracy', 'threshold')
        elif temp_enable is True and (hist_enable is False and pix_enable is False):
            plot_Graph(f1_score_calc_array_t[i], threshold_array_t, max(f1_score_calc_array_t[i]) + 0.1,
                  max(threshold_array_t) + 1,
                  result_folder + 'f1_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f1_score', 'threshold')
            plot_Graph(f2_score_calc_array_t[i], threshold_array_t, max(f2_score_calc_array_t[i]) + 0.1,
                  max(threshold_array_t) + 1,
                  result_folder + 'f2_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f2_score', 'threshold')
            plot_Graph(f05_score_calc_array_t[i], threshold_array_t, max(f05_score_calc_array_t[i]) + 0.1,
                  max(threshold_array_t) + 1,
                  result_folder + 'f05_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'f05_score', 'threshold')
            plot_Graph(precision_array_t[i], recall_array_t[i], max(precision_array_t[i]) + 0.1,
                  max(recall_array_t[0]) + 0.1,
                  result_folder + 'precision_vs_recall_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'recall')
            plot_Graph(sensitivity_array_t[i], specificity_fn_array_t[i], max(sensitivity_array_t[i]) + 0.1,
                  max(specificity_fn_array_t[i]) + 0.1,
                  result_folder + 'sensitivity_vs_specificity_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'specificity')
            plot_Graph(sensitivity_array_t[i], threshold_array_t, max(sensitivity_array_t[i]) + 0.1,
                  max(threshold_array_t) + 1,
                  result_folder + 'sensitivity_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'sensitivity', 'threshold')
            plot_Graph(precision_array_t[i], threshold_array_t, max(precision_array_t[i]) + 0.1, max(threshold_array_t) + 1,
                  result_folder + 'precision_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'precision', 'threshold')
            plot_Graph(accuracy_array_t[i], threshold_array_t, max(accuracy_array_t[i]) + 0.1, max(threshold_array_t) + 1,
                  result_folder + 'accuracy_vs_threshold_t' + '_window_' + str(
                      window_sizes[i]) + '.png', 'accuracy', 'threshold')
    if (hist_enable is True and pix_enable is True) and temp_enable is False:
        plot_Graph(f1_score_calc_array_hp[0], threshold_array_hp, max(f1_score_calc_array_hp[0]) + 0.1,
              max(threshold_array_hp) + 1,
              result_folder + 'f1_vs_threshold_hp' + '.png', 'f1_score',
              'threshold')
        plot_Graph(f2_score_calc_array_hp[0], threshold_array_hp, max(f2_score_calc_array_hp[0]) + 0.1,
              max(threshold_array_hp) + 1,
              result_folder + 'f2_vs_threshold_hp' + '.png', 'f2_score',
              'threshold')
        plot_Graph(f05_score_calc_array_hp[0], threshold_array_hp, max(f05_score_calc_array_hp[0]) + 0.1,
              max(threshold_array_hp) + 1,
              result_folder + 'f05_vs_threshold_hp' + '.png', 'f05_score',
              'threshold')
        plot_Graph(precision_array_hp[0], recall_array_hp[0], max(precision_array_hp[0]) + 0.1,
              max(recall_array_hp[0]) + 0.1,
              result_folder + 'precision_vs_recall_hp' + '.png', 'precision',
              'recall')
        plot_Graph(sensitivity_array_hp[0], specificity_fn_array_hp[0], max(sensitivity_array_hp[0]) + 0.1,
              max(specificity_fn_array_hp[0]) + 0.1,
              result_folder + 'sensitivity_vs_specificity_hp' + '.png',
              'sensitivity', 'specificity')
        plot_Graph(sensitivity_array_hp[0], threshold_array_hp, max(sensitivity_array_hp[0]) + 0.1,
              max(threshold_array_hp) + 1,
              result_folder + 'sensitivity_vs_threshold_hp' + '.png',
              'sensitivity', 'threshold')
        plot_Graph(precision_array_hp[0], threshold_array_hp, max(precision_array_hp[0]) + 0.1, max(threshold_array_hp) + 1,
              result_folder + 'precision_vs_threshold_hp' + '.png',
              'precision', 'threshold')
        plot_Graph(accuracy_array_hp[0], threshold_array_hp, max(accuracy_array_hp[0]) + 0.1, max(threshold_array_hp) + 1,
              result_folder + 'accuracy_vs_threshold_hp' + '.png',
              'accuracy', 'threshold')
    elif hist_enable is True and (pix_enable is False and temp_enable is False):
        plot_Graph(f1_score_calc_array_h[0], threshold_array_h, max(f1_score_calc_array_h[0]) + 0.1,
              max(threshold_array_h) + 1,
              result_folder + 'f1_vs_threshold_h' + '.png', 'f1_score',
              'threshold')
        plot_Graph(f2_score_calc_array_h[0], threshold_array_h, max(f2_score_calc_array_h[0]) + 0.1,
              max(threshold_array_h) + 1,
              result_folder + 'f2_vs_threshold_h' + '.png', 'f2_score',
              'threshold')
        plot_Graph(f05_score_calc_array_h[0], threshold_array_h, max(f05_score_calc_array_h[0]) + 0.1,
              max(threshold_array_h) + 1,
              result_folder + 'f05_vs_threshold_h' + '.png', 'f05_score',
              'threshold')
        plot_Graph(precision_array_h[0], recall_array_h[0], max(precision_array_h[0]) + 0.1, max(recall_array_h[0]) + 0.1,
              result_folder + 'precision_vs_recall_h' + '.png', 'precision',
              'recall')
        plot_Graph(sensitivity_array_h[0], specificity_fn_array_h[0], max(sensitivity_array_h[0]) + 0.1,
              max(specificity_fn_array_h[0]) + 0.1,
              result_folder + 'sensitivity_vs_specificity_h' + '.png',
              'sensitivity', 'specificity')
        plot_Graph(sensitivity_array_h[0], threshold_array_h, max(sensitivity_array_h[0]) + 0.1, max(threshold_array_h) + 1,
              result_folder + 'sensitivity_vs_threshold_h' + '.png',
              'sensitivity', 'threshold')
        plot_Graph(precision_array_h[0], threshold_array_h, max(precision_array_h[0]) + 0.1, max(threshold_array_h) + 1,
              result_folder + 'precision_vs_threshold_h' + '.png',
              'precision', 'threshold')
        plot_Graph(accuracy_array_h[0], threshold_array_h, max(accuracy_array_h[0]) + 0.1, max(threshold_array_h) + 1,
              result_folder + 'accuracy_vs_threshold_h' + '.png', 'accuracy',
              'threshold')
    elif pix_enable is True and (temp_enable is False and hist_enable is False):
        plot_Graph(f1_score_calc_array_p[0], threshold_array_p, max(f1_score_calc_array_p[0]) + 0.1,
              max(threshold_array_p) + 1,
              result_folder + 'f1_vs_phreshold_p' + '.png', 'f1_score',
              'threshold')
        plot_Graph(f2_score_calc_array_p[0], threshold_array_p, max(f2_score_calc_array_p[0]) + 0.1,
              max(threshold_array_p) + 1,
              result_folder + 'f2_vs_phreshold_p' + '.png', 'f2_score',
              'threshold')
        plot_Graph(f05_score_calc_array_p[0], threshold_array_p, max(f05_score_calc_array_p[0]) + 0.1,
              max(threshold_array_p) + 1,
              result_folder + 'f05_vs_phreshold_p' + '.png', 'f05_score',
              'threshold')
        plot_Graph(precision_array_p[0], recall_array_p[0], max(precision_array_p[0]) + 0.1, max(recall_array_p[0]) + 0.1,
              result_folder + 'precision_vs_recall_p' + '.png', 'precision',
              'recall')
        plot_Graph(sensitivity_array_p[0], specificity_fn_array_p[0], max(sensitivity_array_p[0]) + 0.1,
              max(specificity_fn_array_p[0]) + 0.1,
              result_folder + 'sensitivity_vs_specificity_p' + '.png',
              'sensitivity', 'specificity')
        plot_Graph(sensitivity_array_p[0], threshold_array_p, max(sensitivity_array_p[0]) + 0.1, max(threshold_array_p) + 1,
              result_folder + 'sensitivity_vs_phreshold_p' + '.png',
              'sensitivity', 'threshold')
        plot_Graph(precision_array_p[0], threshold_array_p, max(precision_array_p[0]) + 0.1, max(threshold_array_p) + 1,
              result_folder + 'precision_vs_phreshold_p' + '.png',
              'precision', 'threshold')
        plot_Graph(accuracy_array_p[0], threshold_array_p, max(accuracy_array_p[0]) + 0.1, max(threshold_array_p) + 1,
              result_folder + 'accuracy_vs_phreshold_p' + '.png', 'accuracy',
              'threshold')


if __name__ == '__main__':
	main()

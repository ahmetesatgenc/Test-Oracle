features = ['Feature Detector','Descriptor Extractor','Transformation','Image Matching']
keywords_featureDetector = ['FAST','STAR','SIFT','SURF','ORB','BRISK','SimpleBlob']
keywords_descriptorExtractor = ['SIFT','SURF','BRIEF','BRISK','ORB','FREAK']
keywords_transformation = ['Affine Transform','Perspective Transform']
keywords_IM = ['Histogram Matching','Template Matching','Pixel Matching']
image_matching = []*3



def get_Feature():

    config_file = open("conf.txt", "r")
    all_features = config_file.read().split(',')
    index_of_FD = all_features.index(features[0])
    index_of_DE = all_features.index(features[1])
    index_of_T = all_features.index(features[2])
    index_of_IM = all_features.index(features[3])

    for k in range (0,len(keywords_featureDetector)):
        index_of_featureDetector = all_features[index_of_FD:index_of_DE].index(keywords_featureDetector[k])
        if all_features[index_of_FD+index_of_featureDetector+2] == '1':
            feature_detector = keywords_featureDetector[k]
            break
        else:
            feature_detector = None

    for k in range (0,len(keywords_descriptorExtractor)):
        index_of_descriptorExtractor = all_features[index_of_DE:index_of_T].index(keywords_descriptorExtractor[k])
        if all_features[index_of_DE+index_of_descriptorExtractor+2] == '1':
            descriptor_extractor = keywords_descriptorExtractor[k]
            break
        else:
            descriptor_extractor = None

    for k in range (0,len(keywords_transformation)):
        index_of_transformation = all_features[index_of_T:index_of_IM].index(keywords_transformation[k])
        if all_features[index_of_T+index_of_transformation+2] == '1':
            transformation = keywords_transformation[k]
            break
        else:
            transformation = None

    for k in range (0,len(keywords_IM)):
        index_of_imageMatching = all_features[index_of_IM:].index(keywords_IM[k])
        if all_features[index_of_IM+index_of_imageMatching+2] == '1':
            image_matching.append(keywords_IM[k])

    return feature_detector,descriptor_extractor,transformation,image_matching

import numpy as np
import matplotlib.pyplot as plt
from imageMatching import window_sizes

spread_template = []
center_template = []
data_template = []
flier_high_temp = []
flier_low_temp = []
#input_1 ==> array of one of the image matching method result
#input_2==> array of image counts

def box_plot(hist,pixel,template,folder):

    fig1, ax = plt.subplots()

    spread_template = []
    center_template = []
    data_template = []
    if hist is not None:
        spread_hist = hist
        center_hist = [np.median(hist)]* len(hist)
        flier_high_hist = [1]*len(hist)
        flier_low_hist = [0]*len(hist)
        data_histogram = np.concatenate((spread_hist, center_hist, flier_high_hist, flier_low_hist))

    if pixel is not None:
        spread_pixel = pixel
        center_pixel = [np.median(pixel)]*len(pixel)
        flier_high_pixel = [1] * len(pixel)
        flier_low_pixel = [0] * len(pixel)
        data_pixel = np.concatenate((spread_pixel, center_pixel, flier_high_pixel, flier_low_pixel))


    if template is not None:
        for k in range(0,len(window_sizes)):
            flier_high_temp = [max(list(map(list, zip(*template)))[k])] * len(list(map(list, zip(*template)))[k])
            flier_low_temp = [min(list(map(list, zip(*template)))[k])] * len(list(map(list, zip(*template)))[k])
            spread_template.append(list(map(list, zip(*template)))[k])
            center_template.append([np.median(spread_template[k])]*len(list(map(list, zip(*template)))[k]))
            data_template.append(np.concatenate((spread_template[k], center_template[k], flier_high_temp, flier_low_temp)))


    if hist is None and (pixel is not None and template is not None):
        data = [data_pixel]+data_template
        labels = ['Pixel','(640,360)','(480,270)','(320,180)','(160,108)','(120,90)','(16,20)']
    elif pixel is None and (hist is not None and template is not None):
        data = [data_histogram]+data_template
        labels = ['Histogram', '(640,360)', '(480,270)', '(320,180)',
                  '(160,108)', '(120,90)', '(16,20)']
    elif template is None and (pixel is not None and hist is not None):
        data = [data_histogram, data_pixel]
        labels = ['Histogram', 'Pixel']
    elif hist is None and pixel is None and template is not None:
        data = data_template
        labels = ['(640,360)', '(480,270)', '(320,180)',
                  '(160,108)', '(120,90)', '(16,20)']
    elif pixel is None and template is None and hist is not None:
        data = [data_histogram]
        labels = ['Histogram']
    elif hist is None and template is None and pixel is not None:
        data = [data_pixel]
        labels = ['Pixel']
    else:
        data = [data_histogram, data_pixel] + data_template
        labels = ['Histogram','Pixel', '(640,360)', '(480,270)', '(320,180)',
                  '(160,108)', '(120,90)', '(16,20)']

    ax.set_title('Basic Plot')
    #ax.set_xlabel('Image Matching Methods')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticklabels(labels=labels,rotation=45, fontsize=6)
    ax.boxplot(data,labels=labels)
    plt.savefig(folder, dbbox_inches='tight')

    return
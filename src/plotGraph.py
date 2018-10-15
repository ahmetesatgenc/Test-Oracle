import matplotlib.pyplot as plt

def plot_Graph(y_axis, x_axis, y_lim, x_lim, folder, ylabel, xlabel):
    plt.cla()
    plt.figure()
    y = y_axis
    x = x_axis
    plt.plot(x, y,'ro')
    plt.ylim(-1, y_lim)
    plt.xlim(0, x_lim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(folder, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)  # Use fig. here

    return
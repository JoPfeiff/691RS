import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_line_graph(arrays, labels, title_img, x_ticks, x_labels,  colors = ['ro-', 'bo-']): #tuning_parameter,
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ticks = np.arange(0.0, 0.5, 0.02)
    for i in range(len(arrays)):
        array = arrays[i]
        index = range(len(array))
        values = array
        ax.plot(x_ticks, values, colors[i])
        # ax.set_xlabel(x_labels)
        ax.set_ylabel('RMSE/MAE Score')
    plt.title(title_img)
    plt.xticks(x_ticks, x_labels)
    plt.legend(labels, loc = 'right')

    plt.savefig("Figures/"+title_img+".pdf")

    return True

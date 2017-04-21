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



class AxesDecorator():
    def __init__(self, ax, size="5%", pad=0.05, ticks=[1,2,3], spacing=0.05,
                 color="k"):
        self.divider= make_axes_locatable(ax)
        self.ax = self.divider.new_vertical(size=size, pad=pad, sharex=ax, pack_start=True)
        ax.figure.add_axes(self.ax)
        self.ticks=np.array(ticks)
        self.d = np.mean(np.diff(ticks))
        self.spacing = spacing
        self.get_curve()
        self.color=color
        for x0 in ticks:
            self.plot_curve(x0)
        self.ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)
        self.ax.tick_params(axis='x', which=u'both',length=0)
        ax.tick_params(axis='x', which=u'both',length=0)
        for direction in ["left", "right", "bottom", "top"]:
            self.ax.spines[direction].set_visible(False)
        self.ax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel("")
        self.ax.set_xticks(self.ticks)

    def plot_curve(self, x0):
        x = np.linspace(x0-self.d/2.*(1-self.spacing),x0+self.d/2.*(1-self.spacing), 50 )
        self.ax.plot(x, self.curve, c=self.color)

    def get_curve(self):
        lx = np.linspace(-np.pi/2.+0.05, np.pi/2.-0.05, 25)
        tan = np.tan(lx)*10
        self.curve = np.hstack((tan[::-1],tan))
        return self.curve


# # Do your normal plotting
# fig, ax = plt.subplots()
#
# x = [2,3,4,5]
# y = [4,1,3,7]
# ax.scatter(x,y, s=900, c=y, )
# ax.set_ylim([0,10])
# ax.set_xlabel("Strange axis")
#
# #at the end call the AxesDecorator class
# # with the axes as argument
# AxesDecorator(ax, ticks=x)
#
#
# plt.show()
#
# print "Done"
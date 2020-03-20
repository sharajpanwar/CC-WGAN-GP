from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns; sns.set()

# function of creating Heat maps
# data=> target or nonTarget; dir => out put directory; n => title name
def create_heat_map(data, dir, n):
    ax = sns.heatmap(np.squeeze(np.mean(data, axis=0)), cmap="summer")
    plt.title(str(n)+'_Heat_Map')
    plt.savefig(dir + '/' + str(n) + '.png')
    plt.show()

# Function for creating one channel plot
# data=> target or nonTarget; dir => out put directory; ch => channel index; n => title name
def create_one_channel_plot(data, ch, dir, n):
    one_ch_data = np.mean(data[:, ch, :], axis=0)
    plt.plot(one_ch_data)
    plt.title(str(n)+'_plot')
    plt.ylabel('Potential')
    plt.xlabel('Time')
    plt.savefig(dir + '/' + str(n) + '.png')
    plt.show()


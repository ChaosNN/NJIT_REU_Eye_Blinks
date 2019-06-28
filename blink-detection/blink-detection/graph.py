import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#x = []
#y = []
#gy = []

'''
# Data stores data
def frame_data(x1):
    x = x1
    print("Graph Class")
    print(x)


def graph_data(y1):
    # x and y values to be plotted
    y = y1
    print(y)


def ground_data(gy1):
    # ground truth x and y values to be plotted
    gy = gy1
'''


#df = pd.DataFrame({'x': range(1,11), 'foo': np.random.randn(10), 'bar':
    #np.random.randn(10)+range(1,11), 'world': np.random.randn(10)+range(11,21) })

def show(x, y, gy=None):
    # data frame storing the blink information and ground truth
    df = pd.DataFrame({'x': x, "EAR": y, 'Ground Truth': gy})

    # multiple line plot
    # plt.plot('x', 'EAR', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    plt.plot('x', 'EAR', data=df, marker='', color='skyblue', linewidth=4)
    plt.plot('x', 'Ground Truth', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
             label="Ground Truth")
    plt.xlabel("Frame")
    plt.legend()
    plt.show()



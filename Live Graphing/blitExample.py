# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:09:01 2019

@original author: https://learn.sparkfun.com/tutorials/graph-sensor-data-with-python-and-matplotlib/speeding-up-the-plot-animation
@alterations: Miriam Tan
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Parameters
x_len = 35        # Number of points to display
y_range = [0, 10]  # Range of possible Y values to display

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, 35))
ys = [0] * x_len
ax.set_ylim(y_range)

# Create a blank line. We will update the line in animate
line, = ax.plot(xs, ys)

# Add labels
plt.title('Data over Time')
plt.xlabel('Samples')
plt.ylabel('Data')

# This function is called periodically from FuncAnimation
def animate(i, ys):

    # get next data pt
    # right now this generates a random pt
    # eventually switch to the label data
    myDataPt = random.randint(1,9)

    # Add y to list
    ys.append(myDataPt)

    # Limit y list to set number of items
    ys = ys[-x_len:]

    # Update line with new Y values
    line.set_ydata(ys)

    return line,

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig,
    animate,
    fargs=(ys,),
    interval=50,
    blit=True)
plt.show()

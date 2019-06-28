# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 21:51:14 2019

@author: TANMR1
"""

import pandas as pd

# the first and second columns store the frame # and the blink value
# -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)

file_name = 'person2.tag'

'''
# skip the first 19 rows because they don't have data
# and they make the program crash because they appear to have only two columns
# whereas the rest of the data has 20 columns...
Need to find a better/more generalized method...
'''

df = pd.read_csv(file_name,skiprows = 19, sep=':',header=None,skipinitialspace=True)

frame_nums = df.iloc[:,0]
blink_vals = df.iloc[:,1]

'''
# See the keys
#print df.keys()
print df.columns
print("now printing the frame number")
print df.iloc[:, 0]
print("now printing the blink val")
print df.iloc[:, 1]
#print df[1,2]
'''
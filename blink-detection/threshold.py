# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:29:03 2019

@author: TANMR1
"""
import math
'''
find the average of the EAR distance between each pair of consecutive frames
look for
--2 frame section where gap is higher than avg-->would this work bc the avg will
be higher than the non-blink section avgs because of the blink sections?
--3 frame section where gap is higher
--4 frame section where gap is higher
--5 frame section wehre gap is higher

compare to hard coded .2, .25, .3, .35 vals

'''

def frame_to_frame_EAR_diff(EARs):
    prev_ear = EARs[0]
    dists = []
    for ear in EARs:
        dists.append(math.abs(ear-prev_ear))
        prev_ear = ear
    return (mean(dists), dists)

def two_frame_gap_thresh(EARs, avg_dist, dists):
    high_EARs = []
    low_EARs = []
    for frame_idx, dist in enumerate(dists):
        if dist > avg_dist:
            high_EARs.append(max(EARs[frame_idx], EARs[frame_idx-1]))
            low_EARs.append(min(EARs[frame_idx], EARs[frame_idx-1]))
    high_EARs.sort()
    low_EARs.sort()
    top_EARs = high_EARs[int(len(high_EARs) * .9) : int(len(high_EARs) * 1.0)]
    bottom_EARs = low_EARs[int(len(low_EARs) * .9) : int(len(low_EARs) * 1.0)]
    threshold = mean([mean(top_EARs),mean(bottom_EARs))



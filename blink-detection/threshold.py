# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:29:03 2019

@author: TANMR1
"""
import copy
from statistics import mean 
import blink_frame_pairs as bfp
import IOU_eval as evalu


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
        dists.append(abs(ear-prev_ear))
        prev_ear = ear
    return (mean(dists), dists)

def two_frame_gap_thresh(EARs, avg_dist, dists):
    sortedEARs = sorted(EARs)
    high_EARs = []
    low_EARs = []
    for frame_idx, dist in enumerate(dists):
        if dist > avg_dist:
            high_EARs.append(max(sortedEARs[frame_idx], sortedEARs[frame_idx-1]))
            low_EARs.append(min(sortedEARs[frame_idx], sortedEARs[frame_idx-1]))
    high_EARs.sort()
    low_EARs.sort()
    top_EARs = high_EARs[int(len(high_EARs) * .9) : int(len(high_EARs) * 1.0)]
    bottom_EARs = low_EARs[int(len(low_EARs) * 0.0) : int(len(low_EARs) * 0.1)]
    threshold = mean([mean(top_EARs),mean(bottom_EARs)])
    return threshold

def avg_thresh(EARs):
    sortedEARS = sorted(EARs)
    #EARs.sort()
    top_EARs = sortedEARS[int(len(sortedEARS) * .9) : int(len(sortedEARS) * 1.0)]
    bottom_EARs = sortedEARS[int(len(sortedEARS) * 0.0) : int(len(sortedEARS) * 0.1)]
    threshold = mean([mean(top_EARs),mean(bottom_EARs)])
    return threshold

def compare_IOUs(EARs, gt_pairs):
    print("in compare_IOUs of threshold.py")
    print("gt pairs: ", gt_pairs)
    avg_threshold = avg_thresh(EARs)
    pred_pairs_avg = bfp.get_pred_blink_pairs(EARs, avg_threshold)
    print("pred pairs w average threshold: ", pred_pairs_avg)
    IOU_vals_avg_thresh = evalu.IOU_eval(gt_pairs, pred_pairs_avg)
    (avg_dist, dists) = frame_to_frame_EAR_diff(EARs)
    '''
    thresh_2frame = two_frame_gap_thresh(EARs, avg_dist, dists)
    pred_pairs_2frame = bfp.get_pred_blink_pairs(EARs, thresh_2frame)
    print("pred pairs w two frame average threshold: ", pred_pairs_2frame)
    IOU_vals_2frame = evalu.IOU_eval(gt_pairs, pred_pairs_2frame)
    
    pred_pairs_2 = bfp.get_pred_blink_pairs(EARs, .2)
    print("pred pairs w .2 threshold: ", pred_pairs_2)
    IOU_vals_2 = evalu.IOU_eval(gt_pairs, pred_pairs_2)
    
    pred_pairs_25 = bfp.get_pred_blink_pairs(EARs, .25)
    print("pred pairs w .25 threshold: ", pred_pairs_25)
    IOU_vals_25 = evalu.IOU_eval(gt_pairs, pred_pairs_25)
    
    pred_pairs_3 = bfp.get_pred_blink_pairs(EARs, .3)
    print("pred pairs w .3 threshold: ", pred_pairs_3)
    IOU_vals_3 = evalu.IOU_eval(gt_pairs, pred_pairs_3)
    
    pred_pairs_35 = bfp.get_pred_blink_pairs(EARs, .35)
    print("pred pairs w .35 threshold: ", pred_pairs_35)
    IOU_vals_35 = evalu.IOU_eval(gt_pairs, pred_pairs_35)
    
    #print(IOU_vals_avg_thresh, IOU_vals_2frame, IOU_vals_2, IOU_vals_25, IOU_vals_3, IOU_vals_35)
    '''
    
    
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:53:35 2019

@author: TANMR1
"""

'''
GT_blink_vals: an array of 1/-1 blink vals corresponding to each frame
'''
def get_GT_blink_pairs(GT_blink_vals, notblink, blink):
    # the first and second columns store the frame # and the blink value
    # -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)
    GT_blink_pairs = []
    start_frame = 0
    end_frame = 0
    prev = -1
    for frame_idx, blink_val in enumerate(GT_blink_vals):
        if prev == notblink and blink_val == blink:
            start_frame = frame_idx + 1 # +1 to account for 0 indexing
        elif prev == blink and blink_val == notblink:
            end_frame = frame_idx 
            GT_blink_pairs.append([start_frame, end_frame])
            start_frame = 0
            end_frame = 0
        prev = blink_val
    if start_frame != 0 and end_frame == 0:
        GT_blink_pairs.append([start_frame, len(GT_blink_vals)])
    #print("These are the gt blink pairs: ", GT_blink_pairs)
    return GT_blink_pairs

'''
pred_blink_vals: an array of ear vals corresponding to each frame
'''
def get_pred_blink_pairs(pred_blink_vals, EAR_threshold):
    '''
    print("using this threshold to get predicted pairs: ", EAR_threshold)
    print("pred blink vals: ")
    for pred in pred_blink_vals:
        print(pred)
    '''
    pred_blink_pairs = []
    start_frame = 0
    end_frame = 0
    prev = 1
    for frame_idx, blink_val in enumerate(pred_blink_vals):
        '''
        print("prev: ", prev)
        print("start: ", start_frame)
        print("end: ", end_frame)
        print("frame idx: ", frame_idx)
        print("blink val", blink_val)
        '''
        if prev > EAR_threshold and blink_val <= EAR_threshold:
            start_frame = frame_idx + 1 # +1 to account for 0 indexing
        elif prev <= EAR_threshold and blink_val > EAR_threshold:
            end_frame = frame_idx 
            pred_blink_pairs.append([start_frame, end_frame])
            start_frame = 0
            end_frame = 0
        prev = blink_val
    if start_frame != 0 and end_frame == 0:
        pred_blink_pairs.append([start_frame, len(pred_blink_vals)])
    #print("These are the predicted blink pairs: ", pred_blink_pairs)
    return pred_blink_pairs

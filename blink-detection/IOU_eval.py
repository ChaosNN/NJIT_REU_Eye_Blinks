# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:25:36 2019

@author: TANMR1
"""

'''
GT_blink_vals: an array of frame numbers and corresponding 1/-1 blink vals
'''
def get_GT_blink_pairs(GT_blink_vals):
    # the first and second columns store the frame # and the blink value
    # -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)
    GT_blink_pairs = []
    start_frame = 0
    end_frame = 0
    prev = -1
    for frame_idx, blink_val in enumerate(GT_blink_vals):
        if prev == -1 and blink_val == 1:
            start_frame = frame_idx
        elif prev == 1 and blink_val == -1:
            end_frame = frame_idx - 1
            GT_blink_pairs.append([start_frame, end_frame])
            start_frame = 0
            end_frame = 0
    if start_frame != 0 and end_frame == 0:
        end_frame = frame_idx - 1
        GT_blink_pairs.append([start_frame, end_frame])
    print GT_blink_pairs
    return GT_blink_pairs

'''
GT_blinks: array of pairs for the ground truth blinks
pred_blinks: array of pairs for the predicted blinks
----each pair is the starting and ending frame number of a blink
'''
def IOU_eval(GT_blinks, pred_blinks):
    # intersect over union of ground truth vs prediction blink frames evaluation method
    # considered in A. Fogelton, W. Benesova's Computer Vision and Image Understanding (2016)
    iou_threshold = 0.2
    g_idx = 0
    p_idx = 0
    TP_Counter = 0
    FP_Counter = 0
    FN_Counter = 0
    GT_blinks = []
    pred_blinks = []
    while g_idx < GT_blinks.size and p_idx < pred_blinks.size:
        
        GT_start_frame = GT_blinks(g_idx).start
        GT_end_frame = GT_blinks(g_idx).end
        pred_start_frame = pred_blinks(p_idx).start
        pred_end_frame = pred_blinks(p_idx).end
        # the ground truth and prediction overlap: so find the iou
        # find the intersect and union of the groundtruth and prediction blink frames
        GT_pred_union = max(GT_end_frame, pred_end_frame) - min(GT_start_frame, pred_start_frame)
        GT_pred_intersect = min(GT_end_frame, pred_end_frame) - max(GT_start_frame, pred_start_frame)
        iou = GT_pred_intersect / GT_pred_union

        if iou > iou_threshold:
            TP_Counter += 1
            p_idx += 1
            g_idx += 1
        elif pred_end_frame < GT_end_frame:
            FP_Counter += 1
            p_idx += 1
        else:
            FN_Counter += 1
            g_idx += 1
    FP_Counter += pred_blinks.size - p_idx
    FN_Counter += GT_blinks.size - g_idx
    
    return (FP_Counter, FN_Counter, TP_Counter)

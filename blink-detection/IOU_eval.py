# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:25:36 2019

@author: TANMR1
"""
# USAGE
# python IOU_eval.py

import blink_frame_pairs as bfp

def get_recall(TP, FN):
    return TP/(TP + FN)
    
def get_precision(TP, FP):
    return TP/(TP + FP)
    
def evaluate(EARs, gt_blinks, thresholds, EYE_AR_THRESH):
    results = []
    gt_pairs = bfp.get_GT_blink_pairs(gt_blinks, EYE_AR_THRESH)
    for threshold in thresholds:
        pred_pairs = bfp.get_pred_blink_pairs(EARs, threshold)
        (FP, FN, TP) = IOU_eval(gt_pairs, pred_pairs)
        recall = get_recall(TP, FN)
        precision = get_precision(TP, FP)
        results.append((threshold, pred_pairs, gt_pairs, FP, FN, TP, recall, precision))
    return results


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
    while g_idx < len(GT_blinks) and p_idx < len(pred_blinks):
        
        GT_start_frame = GT_blinks[g_idx][0]
        GT_end_frame = GT_blinks[g_idx][1]
        pred_start_frame = pred_blinks[p_idx][0]
        pred_end_frame = pred_blinks[p_idx][1]
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
    #print(FP_Counter, FN_Counter, TP_Counter)
    FP_Counter += len(pred_blinks) - p_idx
    FN_Counter += len(GT_blinks) - g_idx
    
    return (FP_Counter, FN_Counter, TP_Counter)

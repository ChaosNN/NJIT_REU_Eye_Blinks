# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:25:36 2019

@author: TANMR1
"""


def IOU_eval():
    # intersect over union of ground truth vs prediction blink frames evaluation method
    # considered in A. Fogelton, W. Benesova's Computer Vision and Image Understanding (2016)
    iou_threshold = 0.2
    g = 0
    p = 0
    TP_Counter = 0
    FP_Counter = 0
    FN_Counter = 0
    GT_blinks = []
    pred_blinks = []
    while g < GT_blinks.size and p < pred_blinks.size:
        
        GT_start_frame = GT_blinks(g).start
        GT_end_frame = GT_blinks(g).end
        pred_start_frame = pred_blinks(p).start
        pred_end_frame = pred_blinks(p).end
        # the ground truth and prediction overlap: so find the iou
        # find the intersect and union of the groundtruth and prediction blink frames
        GT_pred_union = max(GT_end_frame, pred_end_frame) - min(GT_start_frame, pred_start_frame)
        GT_pred_intersect = min(GT_end_frame, pred_end_frame) - max(GT_start_frame, pred_start_frame)
        iou = GT_pred_intersect / GT_pred_union

        if iou > iou_threshold:
            TP_Counter += 1
            p += 1
            g += 1
        elif pred_end_frame < GT_end_frame:
            FP_Counter += 1
            p += 1
        else:
            FN_Counter += 1
            g += 1
    FP_Counter += pred_blinks.size - p
    FN_Counter += GT_blinks.size - g
    
    return (FP_Counter, FN_Counter, TP_Counter)

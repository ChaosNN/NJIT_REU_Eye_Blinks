# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:25:36 2019

@author: TANMR1
"""
# USAGE
# python IOU_eval.py


# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os


'''
GT_blink_vals: an array of 1/-1 blink vals corresponding to each frame
'''
def get_GT_blink_pairs(GT_blink_vals):
    # the first and second columns store the frame # and the blink value
    # -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)
    GT_blink_pairs = []
    start_frame = 0
    end_frame = 0
    prev = -1
    for frame_idx, blink_val in enumerate(GT_blink_vals):
        if prev == -1 and blink_val > 0:
            start_frame = frame_idx
        elif prev > 0 and blink_val == -1:
            end_frame = frame_idx - 1
            GT_blink_pairs.append([start_frame, end_frame])
            start_frame = 0
            end_frame = 0
        prev = blink_val
    if start_frame != 0 and end_frame == 0:
        end_frame = frame_idx - 1
        GT_blink_pairs.append([start_frame, end_frame])
    print GT_blink_pairs
    return GT_blink_pairs

'''
pred_blink_vals: an array of ear vals corresponding to each frame
'''
def get_pred_blink_pairs(pred_blink_vals, EAR_threshold):
    pred_blink_pairs = []
    start_frame = 0
    end_frame = 0
    prev = -1
    for frame_idx, blink_val in enumerate(pred_blink_vals):
        if prev > EAR_threshold and blink_val <= EAR_threshold:
            start_frame = frame_idx
        elif prev <= EAR_threshold and blink_val > EAR_threshold:
            end_frame = frame_idx - 1
            pred_blink_pairs.append([start_frame, end_frame])
            start_frame = 0
            end_frame = 0
    if start_frame != 0 and end_frame == 0:
        end_frame = frame_idx - 1
        pred_blink_pairs.append([start_frame, end_frame])
    print pred_blink_pairs
    return pred_blink_pairs

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


def main():
    '''
    read_data('zju')
    num_rows = df_videodata.shape[0]
    for i in range(num_rows):
        video_filename = get_VIDEO_FILENAME(i)
        tag_filename = get_TAG_FILENAME(i)
        png_filename = get_PNG_FILENAME(i)
        path = get_PATH(i)
        gt_blinks = get_GT_blinks(tag_filename)
        (detector, predictor, lStart, lEnd, rStart, rEnd) = init_detector_predictor()
        (vs, fileStream) = start_videostream(video_filename)
        EARs = scan_and_display_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd)
        # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
        graph_EAR_GT(EARs, gt_blinks, path, png_filename)
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
    '''
    path = ''
    video_filename = '000001M_FBN.avi'
    tag_filename = '000001M_FBN.tag'
    png_filename = '000001M_FBN.png'
    gt_blinks = get_GT_blinks(tag_filename)
    (detector, predictor, lStart, lEnd, rStart, rEnd) = init_detector_predictor()
    (vs, fileStream) = start_videostream(video_filename)
    EARs = scan_and_display_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd)
    # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
    graph_EAR_GT(EARs, gt_blinks, path, png_filename)   
    print("finished graphing")     
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    print("post cleanup")
    '''

if __name__ == '__main__':
    main()

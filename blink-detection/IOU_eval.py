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

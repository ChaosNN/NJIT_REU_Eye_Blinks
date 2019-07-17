# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:29:03 2019

@author: TANMR1
"""
import copy
from statistics import mean 
import statistics
import blink_frame_pairs as bfp
import IOU_eval as evalu
import save_results as save

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
    try:
        threshold = mean([mean(top_EARs),mean(bottom_EARs)])
    except statistics.StatisticsError:
        threshold = mean([mean(high_EARs), mean(low_EARs)])
    if threshold > .35 or threshold < .15:
        print("Danger. Two Frame Gap EAR Threshold value = ", threshold )
    return threshold

def avg_thresh(EARs):
    sortedEARS = sorted(EARs)
    #EARs.sort()
    top_EARs = sortedEARS[int(len(sortedEARS) * .9) : int(len(sortedEARS) * 1.0)]
    bottom_EARs = sortedEARS[int(len(sortedEARS) * 0.0) : int(len(sortedEARS) * 0.1)]
    try:
        threshold = mean([mean(top_EARs),mean(bottom_EARs)])
    except statistics.StatisticsError:
        threshold = mean(EARs)
    if threshold > .35 or threshold < .15:
        print("Danger. Avg EAR Threshold value = ", threshold )
    return threshold

def compare_IOUs(EARs, bool_gt, gt_pairs, file_path, file):
    #print("in compare_IOUs of threshold.py")
    #print("gt pairs: ", gt_pairs)
    frame_ct = len(EARs)
    ga_t = avg_thresh(EARs)
    pbp_gat = bfp.get_pred_blink_pairs(EARs, ga_t)
    pred_ga = bfp.get_blink(pbp_gat, frame_ct)
    #print("pred pairs w average threshold: ", pred_pairs_avg)
    (tp_gat, fp_gat, fn_gat, prec_gat, recall_gat) = evalu.IOU_eval(gt_pairs, pbp_gat)
    
    (avg_dist, dists) = frame_to_frame_EAR_diff(EARs)
    
    tcf_t = two_frame_gap_thresh(EARs, avg_dist, dists)
    pbp_tcft = bfp.get_pred_blink_pairs(EARs, tcf_t)
    pred_tcf = bfp.get_blink(pbp_tcft, frame_ct)
    #print("pred pairs w two frame average threshold: ", pred_pairs_2frame)
    (tp_tcft, fp_tcft, fn_tcft, prec_tcft, recall_tcft) = evalu.IOU_eval(gt_pairs, pbp_tcft)
    
    h15_t = 0.15
    pbp_h15t = bfp.get_pred_blink_pairs(EARs, h15_t)
    pred_h15 = bfp.get_blink(pbp_h15t, frame_ct)
    #print("pred pairs w .2 threshold: ", pbp_h2t)
    (tp_h15t, fp_h15t, fn_h15t, prec_h15t, recall_h15t) = evalu.IOU_eval(gt_pairs, pbp_h15t)
    
    h2_t = 0.2
    pbp_h2t = bfp.get_pred_blink_pairs(EARs, h2_t)
    pred_h2 = bfp.get_blink(pbp_h2t, frame_ct)
    #print("pred pairs w .2 threshold: ", pbp_h2t)
    (tp_h2t, fp_h2t, fn_h2t, prec_h2t, recall_h2t) = evalu.IOU_eval(gt_pairs, pbp_h2t)
    
    h25_t = 0.25
    pbp_h25t = bfp.get_pred_blink_pairs(EARs, h25_t)
    pred_h25 = bfp.get_blink(pbp_h25t, frame_ct)
    #print("pred pairs w .25 threshold: ", pbp_h25t)
    (tp_h25t, fp_h25t, fn_h25t, prec_h25t, recall_h25t) = evalu.IOU_eval(gt_pairs, pbp_h25t)
    
    h3_t = 0.3
    pbp_h3t = bfp.get_pred_blink_pairs(EARs, h3_t)
    pred_h3 = bfp.get_blink(pbp_h3t, frame_ct)
    #print("pred pairs w .3 threshold: ", pbp_h3t)
    (tp_h3t, fp_h3t, fn_h3t, prec_h3t, recall_h3t) = evalu.IOU_eval(gt_pairs, pbp_h3t)
    
    h35_t = 0.35
    pbp_h35t = bfp.get_pred_blink_pairs(EARs, h35_t)
    pred_h35 = bfp.get_blink(pbp_h35t, frame_ct)
    #print("pred pairs w .35 threshold: ", pbp_h35t)
    (tp_h35t, fp_h35t, fn_h35t, prec_h35t, recall_h35t) = evalu.IOU_eval(gt_pairs, pbp_h35t)
    '''
    graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
    graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
    graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
    graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
    graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
    '''
    save.save_csv(EARs, bool_gt, gt_pairs, 
             ga_t, pred_ga, pbp_gat, tp_gat, fp_gat, fn_gat, prec_gat, recall_gat, 
             tcf_t, pred_tcf, pbp_tcft, tp_tcft, fp_tcft, fn_tcft, prec_tcft, recall_tcft, 
             h15_t, pred_h15, pbp_h15t, tp_h15t, fp_h15t, fn_h15t, prec_h15t, recall_h15t,
             h2_t, pred_h2, pbp_h2t, tp_h2t, fp_h2t, fn_h2t, prec_h2t, recall_h2t, 
             h25_t, pred_h25, pbp_h25t, tp_h25t, fp_h25t, fn_h25t, prec_h25t, recall_h25t, 
             h3_t, pred_h3, pbp_h3t, tp_h3t, fp_h3t, fn_h3t, prec_h3t, recall_h3t, 
             h35_t, pred_h35, pbp_h35t, tp_h35t, fp_h35t, fn_h35t, prec_h35t, recall_h35t, 
             file_path, file)
    
    
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:45:29 2019
@author: TANMR1
"""

# USAGE
# python noargs_multigraph.py

# import the necessary packages
import cv2
import pandas as pd
import os
import IOU_eval as evalu
import blink_frame_pairs as bfp
import start_vid_analyzing as vid
import save_results as save
import threshold as thresh

'''
GLOBAL VARS
'''

# eye aspect ratio to indicate blink 
EYE_AR_THRESH = 0.3
column_name = ['video_file_avi', 'dat_file', 'text_file', 'path', 'png_file', 'folder']
df_videodata = pd.DataFrame(columns=column_name)


# gets the information about the file paths of the selected dataset
def read_data(dataset_name):
    mypath = os.path.join(os.getcwd(), 'data_sets\\', dataset_name)
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        #print(filenames)
        if not filenames:
            print("empty")
        if filenames:
            filenames.append(dirpath)
            filenames.append(filenames[0][:-3] + 'png')
            dir_split = dirpath.split('\\')
            dir_split = dir_split[len(dir_split) - 1]
            filenames.append(dir_split)
            df_videodata.loc[len(df_videodata)] = filenames
            df_videodata[column_name] = df_videodata[column_name].astype(str)

    return df_videodata

def get_VIDEO_FILENAME(i):
      return df_videodata.at[i, 'video_file_avi']


def get_TAG_FILENAME(i):
    return os.path.join(df_videodata.iloc[i]['path'], df_videodata.iloc[i]['dat_file'])


def get_PNG_FILENAME(i):
    return df_videodata.at[i, 'png_file']


def get_PATH(i):
    return df_videodata.at[i, 'path']


def get_FOLDERNAME(i):
    return df_videodata.at[i, 'folder']


def get_GT_blinks(tag_filename):
    # the first and second columns store the frame # and the blink value
    # -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)
    mypath = os.path.join(os.getcwd(), 'data_sets\\', tag_filename)
    rows_to_skip = 0
    # find the number of headerlines to be skipped (varies file to file)
    searchfile = open(mypath, "r")
    for i, line in enumerate(searchfile):
        if "#start" in line: rows_to_skip = i + 1
    searchfile.close()
    df = pd.read_csv(mypath, skiprows= rows_to_skip, sep=':', header=None, skipinitialspace=True)
    blink_vals = (df.iloc[:, 1]).replace(-1, 0)
    blink_vals = (blink_vals).mask(blink_vals > 0, EYE_AR_THRESH)
    return blink_vals

def main():
    
    read_data('zju')
    num_rows = df_videodata.shape[0]

    for i in range(num_rows):
        video_filename = get_VIDEO_FILENAME(i)
        print(video_filename)
        tag_filename = get_TAG_FILENAME(i)
        png_filename = get_PNG_FILENAME(i)
        path = get_PATH(i)
        video_filename = os.path.join(path, video_filename)
        gt_blinks = get_GT_blinks(tag_filename)
        (detector, predictor, lStart, lEnd, rStart, rEnd) = vid.init_detector_predictor()
        (vs, fileStream) = vid.start_videostream(video_filename)
        EARs = vid.start_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd, EYE_AR_THRESH)
        #pred_pairs = bfp.get_pred_blink_pairs(EARs, EYE_AR_THRESH)
        gt_pairs = bfp.get_GT_blink_pairs(gt_blinks, 0.0, 0.3)
        '''
        (TP, FP, FN) = evalu.IOU_eval(gt_pairs, pred_pairs)
        recall = evalu.get_recall(len(TP), len(FN))
        precision = evalu.get_precision(len(TP), len(FP))
        print(gt_pairs, pred_pairs, recall, precision)
        '''
        # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
        folder = get_FOLDERNAME(i)
        print(folder)
        (file_path, file) = save.check_path(path,folder)       
        save.graph_EAR_GT(EARs, gt_blinks, png_filename, file_path, file)
        thresh.compare_IOUs(EARs, gt_pairs, file_path, file)
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
        

    '''
    path = ''
    video_filename = '000001M_FBN.mp4'
    tag_filename = '000001M_FBN.tag'
    png_filename = '000001M_FBN.png'
    gt_blinks = get_GT_blinks(tag_filename)
    (detector, predictor, lStart, lEnd, rStart, rEnd) = vid.init_detector_predictor()
    (vs, fileStream) = vid.start_videostream(video_filename)
    EARs = vid.start_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd, EYE_AR_THRESH)
    print("original ears: ")
    for ear in EARs:
        print(ear)
    # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
    gt_pairs = bfp.get_GT_blink_pairs(gt_blinks, 0.0, 0.3)
    thresh.compare_IOUs(EARs, gt_pairs)
    graph.graph_EAR_GT(EARs, gt_blinks, path, png_filename)   
    print("finished graphing")     
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    print("post cleanup")
    '''

if __name__ == '__main__':
    main()

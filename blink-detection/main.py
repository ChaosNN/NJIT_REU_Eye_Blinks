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
#import IOU_eval as evalu
#import blink_frame_pairs as bfp
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


def main():
    
    read_data('zju')
    num_rows = df_videodata.shape[0]

    for i in range(num_rows):
        video_filename = get_VIDEO_FILENAME(i)
        print(video_filename)
        png_filename = get_PNG_FILENAME(i)
        folder = get_FOLDERNAME(i)
        path = get_PATH(i)
        print("folder name: ", folder)
        video_filename = os.path.join(path, video_filename)
        (detector, predictor, lStart, lEnd, rStart, rEnd) = vid.init_detector_predictor()
        (vs, fileStream) = vid.start_videostream(video_filename)
        EARs = vid.better_start_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd, EYE_AR_THRESH)

        (file_path, file) = save.check_path(path, folder)
        save.graph_EAR(EARs, png_filename, file_path, file)
        thresh.compare_IOUs(EARs, file_path, file)

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == '__main__':
    main()

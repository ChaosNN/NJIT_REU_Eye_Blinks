# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:45:29 2019
@author: TANMR1
"""

# USAGE
# python noargs_multigraph.py

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
GLOBAL VARS
'''

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3    
SHAPE_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

df_videodata = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path', 'png_file', 'folder'])


# gets the information about the file paths of the selected dataset
def read_data(dataset_name):
    mypath = os.path.join(os.getcwd(), 'data_sets\\', dataset_name)
    print(mypath)
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        if not filenames:
            print("empty")
        if filenames:
            filenames.append(dirpath)
            filenames.append(filenames[0][:-3] + 'png')
            dir_split = dirpath.split('\\')
            dir_split = dir_split[len(dir_split) - 1]
            filenames.append(dir_split)
            df_videodata.loc[len(df_videodata)] = filenames

    return df_videodata

# saves the results of the calculations in a csv file
# result is a string variable containing the path to the data set result's directory
'''
def results_file(result):
    result_file = os.path.join(os.getcwd(), 'data_sets\\', dataset_name + '_results', file_name + '.csv')
    df_videodata.to_csv(result_file)
'''

# checks if directory exists, if not the directory is constructed
def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# checks if the csv file exists
def check_file(file_path):
    return os.path.isfile(file_path)


# creates a data frame, which is saved as a csv file
def save_csv(file_path, ear, blink):
    df = pd.DataFrame(ear, columns=['EAR'])
    df['Ground Truth'] = pd.Series(blink)
    df.to_csv(file_path)

def get_VIDEO_FILENAME(i):
    return df_videodata.at[i, 'video_file']

def get_TAG_FILENAME(i):
    return os.path.join(df_videodata.iloc[i]['path'], df_videodata.iloc[i]['dat_file'])
    #return df_videodata.at[i, 'dat_file']


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
    #mypath = tag_filename
    rows_to_skip = 0
    # find the number of headerlines to be skipped (varies file to file)
    searchfile = open(mypath, "r")
    for i, line in enumerate(searchfile):
        if "#start" in line: rows_to_skip = i + 1
    searchfile.close()
    df = pd.read_csv(mypath, skiprows= rows_to_skip, sep=':', header=None, skipinitialspace=True)
    blink_vals = (df.iloc[:, 1]).replace(-1, 0)
    blink_vals = (blink_vals).mask(blink_vals > 0, 0.35)
    return blink_vals


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def init_detector_predictor():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILENAME)
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    return (detector, predictor, lStart, lEnd, rStart, rEnd)


def start_videostream(video_filename):
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(video_filename).start()
    fileStream = True
    time.sleep(1.0)
    return (vs, fileStream)


def scan_and_display_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd):
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    EARs = []
    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break
            # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        frame = vs.read()
        if np.shape(frame) != ():
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # Set up plot to call animate() function periodically
            EARs.append(ear)
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        if np.shape(frame) != ():
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    return EARs


def scan_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd):
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    EARs = []
    # loop over frames from the video stream
    while True:
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process
        if fileStream and not vs.more():
            break
            # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale channels)
        frame = vs.read()
        if np.shape(frame) != ():
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
        rects = detector(gray, 0)
        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            # Set up plot to call animate() function periodically
            EARs.append(ear)
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    # reset the eye frame counter
                COUNTER = 0
    return EARs

def graph_EAR_GT(EARs, blink_vals, path, png_filename, folder):
    plt.xlabel('Frame Number')
    plt.ylabel('EAR')
    plt.plot(EARs, 'b')
    plt.plot(blink_vals, 'r')
    # C:\Users\peted\Documents\Git_Hub\NJIT_REU_Eye_Blinks\blink-detection\data_sets\zju_results
    #file = os.path.join(path, png_filename)
    file = os.path.join(os.getcwd(), 'data_sets\\', 'zju_results\\', folder)
    result = 'results' + folder
    path = os.path.join(file, result)
    try:
        check_dir(file)
        #check_file(file)
        check_file(path + '.csv')
        #os.path.exists(file)

    except IOError:
        print("File exists and will be overwritten")
    finally:
        save_csv(os.path.join(path + '.csv'), EARs, blink_vals)
        #plt.savefig(os.path.join(file, result), bbox_inches='tight')
        plt.savefig(path, bbox_inches='tight')

    plt.close()


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
    print("in gt_blink_pairs")
    for frame_idx, blink_val in enumerate(GT_blink_vals):
        #print("prev: ", prev)
        #print("BLINK_VAL: ", blink_val)
        if prev == -1 and blink_val == 1:
            print("SETTING START FRAME")
            start_frame = frame_idx
        elif prev == 1 and blink_val == -1:
            end_frame = frame_idx - 1
            GT_blink_pairs.append([start_frame, end_frame])
            '''
            print("prev: ", prev)
            print("start: ", start_frame)
            print("end: ", end_frame)
            '''
            start_frame = 0
            end_frame = 0
        prev = blink_val
    if start_frame != 0 and end_frame == 0:
        end_frame = frame_idx - 1
        GT_blink_pairs.append([start_frame, end_frame])
    print(GT_blink_pairs)
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
            '''
            print("prev: ", prev)
            print("start: ", start_frame)
            print("end: ", end_frame)
            '''
            start_frame = 0
            end_frame = 0
    if start_frame != 0 and end_frame == 0:
        end_frame = frame_idx - 1
        pred_blink_pairs.append([start_frame, end_frame])
    print(pred_blink_pairs)
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
    while g_idx < GT_blinks.len and p_idx < pred_blinks.len:
        
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
        pred_pairs = get_pred_blink_pairs(EARs, EYE_AR_THRESH)
        gt_pairs = get_GT_blink_pairs(gt_blinks)
        IOU_vals = IOU_eval(pred_pairs, gt_pairs)
        print(IOU_vals)
        # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
        folder = get_FOLDERNAME(i)
        graph_EAR_GT(EARs, gt_blinks, path, png_filename, folder)
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

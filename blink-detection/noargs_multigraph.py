# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:45:29 2019
@author: TANMR1
"""

# USAGE
# python graph_retry.py --shape-predictor shape_predictor_68_face_landmarks.dat --video 000001M_FBN.mp4

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

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

VIDE0_FILENAME = '' #'000001M_FBN.mp4'
TAG_FILENAME = '000001M_FBN.tag'
# What is this file doing? Do we need this for every eye blink detection algorithm?
SHAPE_PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"

df_videodata = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path'])


# gets the information about the file paths of the selected dataset
def read_data():
    data_set = 'data_sets\zju'
    mypath = os.path.join(os.getcwd(), data_set)
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        if not filenames:
            print("empty")
        if filenames:
            filenames.append(dirpath)
            df_videodata.loc[len(df_videodata)] = filenames

    #print(df_videodata.shape[0])
    print(mypath)


def get_VIDEO_FILENAME(i):
    read_data()
    return df_videodata.at[i, 'video_file']


# returns the tag data file
def get_TAG_FILENAME(i):
    return df_videodata.at[i, 'dat_file']

def get_GT_blinks():
    # the first and second columns store the frame # and the blink value
    # -1 = no blink, all other numbers tell which blink you're on (e.g. 1,2,3,...)
    df = pd.read_csv(TAG_FILENAME, skiprows=19, sep=':', header=None, skipinitialspace=True)
    frame_nums = df.iloc[:, 0]
    blink_vals = (df.iloc[:, 1]).replace(-1, 0)
    blink_vals = (blink_vals).mask(blink_vals > 0, 0.3)
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


def start_videostream():
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = FileVideoStream(VIDE0_FILENAME).start()
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


def graph_EAR_GT(EARs, blink_vals):
    # Add labels
    plt.xlabel('Frame Number')
    plt.ylabel('EAR')
    # frames = max(EARs.count, blink_vals.count)
    fig1 = plt.figure()
    plt.plot(EARs, 'b')
    plt.show()
    fig2 = plt.figure()
    plt.plot(blink_vals, 'r')
    plt.show()
    # plt.plot( 'x', 'y1', data=EARs, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot( 'x', 'y2', data=blink_vals, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
    # plt.legend()
    plt.pause(15)

'''
def IOU_eval():
    # intersect over union of ground truth vs prediction blink frames evaluation method
    # considered in A. Fogelton, W. Benesova's Computer Vision and Image Understanding (2016)
    iou_threshold = 0.2
    TP_Counter = 0
    FP_Counter = 0
    FN_Counter = 0
    blinkpairs = []
    for blink in blinkpairs:
        GT_start_frame =
        GT_end_frame =
        pred_start_frame =
        pred_end_frame =
        # find the intersect and union of the groundtruth and prediction blink frames
        GT_pred_union = max(GT_end_frame, pred_end_frame) - min(GT_start_frame, pred_start_frame)
        GT_pred_intersect = min(GT_end_frame, pred_end_frame) - max(GT_start_frame, pred_start_frame)
        iou = GT_pred_intersect / GT_pred_union

        if iou > iou_threshold:
            TP_Counter += 1
        else:
            FP_Counter += 1
            FN_Counter += 1
'''


def main():
    gt_blinks = get_GT_blinks()
    (detector, predictor, lStart, lEnd, rStart, rEnd) = init_detector_predictor()
    (vs, fileStream) = start_videostream()
    EARs = scan_and_display_video(fileStream, vs, detector, predictor, lStart, lEnd, rStart, rEnd)
    # EARs = scan_video(fileStream, vs, detector, predictor,lStart,lEnd, rStart, rEnd)
    graph_EAR_GT(EARs, gt_blinks)
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()



if __name__ == '__main__':
    main()

from __future__ import print_function
from openCV_Test import PhotoBoothApp
from imutils.video import VideoStream
import argparse
import time

videoFile = "C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection/000001M_FBN.mp4"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
#args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] " + videoFile)
# to use a Raspberry Pi camera
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
vs = VideoStream(videoFile).start()
print("[INOF] starting video " + videoFile)
time.sleep(2.0)

#start the app
#pba = PhotoBoothApp(vs, args["output"])
pba = PhotoBoothApp(vs, "C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection")
pba.root.mainloop()
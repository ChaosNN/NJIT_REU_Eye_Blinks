'''
# https://www.pyimagesearch.com/2016/05/30/displaying-a-video-feed-with-opencv-and-tkinter/
from __future__ import print_function
from PIL import Image, ImageTk
import tkinter as tki
import threading
import datetime
import time
import imutils
from imutils.video import FileVideoStream
import cv2
import os



class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # start the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the tread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot",
                         command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10, pady=10)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        # super ugly hack to get around a runt time error most likely due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                #self.frame = imutils.resize(self.frame, width=300)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        #except RuntimeError, e:
        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    # will need to convert to updating a blink not automatically labeled
    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, self.frame.copy())
        print_function("[INFO] saved {}", format(filename))

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


videoFile = "C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection/000001M_FBN.mp4"
output = "C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection"
#cap = cv2.VideoCapture(0).start()

def start_videostream(video_filename):
    # start the video stream thread
    #print("[INFO] starting video stream thread...")
    vs = FileVideoStream(video_filename).start()
    #fileStream = True
    time.sleep(1.0)
    return (vs)
    #return (vs, fileStream)

PhotoBoothApp(start_videostream(videoFile), output)

tki.mainloop()
'''

'''
#https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class App:
    def __init__(self, window, window_title, video_source="C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection/000001M_FBN.mp4"):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        root = tk.Tk()

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        #self.canvas1 = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        #self.canvas1.pack()

        figure = plt.Figure(figsize=(6,5), dpi=100)
        ax = figure.add_subplot(111)
        chart_type = FigureCanvasTkAgg(figure, root)
        self.canvas1 = chart_type.get_tk_widget()
        ax.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])
        self.canvas1.pack(anchor="s", expand=True)
        
        #canvas1 = FigureCanvasTkAgg(f, self)
        #canvas1.show()
        #canvas1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        


        #self.canvas1 = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        #self.canvas1.pack()


        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
           self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
           self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source="C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection/000001M_FBN.mp4"):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV")
'''

import cv2

cap = cv2.VideoCapture("C:/Users/peted/Documents/Git_Hub/NJIT_REU_Eye_Blinks/blink-detection/000001M_FBN.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
    err, img = cap.read()
    pass

cv2.namedWindow('Frame by Frame')
cv2.createTrackbar('start', 'Frame by Frame', 0, length, onChange)
cv2.createTrackbar('end', 'Frame by Frame', 100, length, onChange)

onChange(0)
cv2.waitKey()

start = cv2.createTrackbarPos('start', 'Frame by Frame')
end = cv2.createTrackbarPos('end', 'Frame by Frame')

if start >= end:
    raise Exception("start must be less than end")

cap.set(cv2.CAP_PROP_POS_FRAMES, start)

while cap.isOpened():
    err, img = cap.read()
    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
        break
    cv2.imshow('Frame by Frame', img)
    k = cv2.waitKey(10) & 0xff
    if k==27:
        break

'''
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
'''
'''
import tkinter
import PIL.Image, PIL.ImageTk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import cv2
import time


'''
'''
root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)


canvas.mpl_connect("key_press_event", on_key_press)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
'''
'''
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas2 = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack(side="top", expand=False, fill="both")
        self.canvas2.pack(sid="top", expand=False, fill="both")

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        print(frame)
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NE)
            self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
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
App(tkinter.Tk(), "Tkinter and OpenCV")
'''


'''
from tkinter import *

root = Tk()
root.geometry("600x400")

top = Frame(root, borderwidth=2, relief="solid")
bottom = Frame(root, borderwidth=2, relief="solid")
container = Frame(top, borderwidth=2, relief="solid")
box1 = Frame(bottom, borderwidth=2, relief="solid")
box2 = Frame(bottom, borderwidth=2, relief="solid")

canvas11 = Canvas(top)
canvas12 = Canvas(bottom)
#label1 = Label(container, text="I could be a canvas, but I'm a label right now")
label2 = Label(top, text="I could be a button")
label3 = Label(top, text="So could I")
label4 = Label(box1, text="I could be your image")
label5 = Label(box2, text="I could be your setup window")

top.pack(side="top", expand=True, fill="both")
bottom.pack(side="bottom", expand=True, fill="both")
container.pack(expand=True, fill="both", padx=5, pady=5)
box1.pack(expand=True, fill="both", padx=10, pady=10)
box2.pack(expand=True, fill="both", padx=10, pady=10)

canvas11.pack()
canvas12.pack()
#label1.pack()
label4.pack()
label5.pack()
label2.pack()
label3.pack()

root.mainloop()
'''

'''
import tkinter
import PIL.Image
import PIL.ImageTk
import cv2


class App:
    def __init__(self, window, video_source1, video_source2):
        self.window = window
        self.window.title("KEC MEDIA PLAYER")
        self.video_source1 = video_source1
        self.video_source2 = video_source2
        self.photo1 = ""
        self.photo2 = ""

        # open video source
        self.vid1 = MyVideoCapture(self.video_source1, self.video_source2)

        # Create a canvas that can fit the above video source size
        self.canvas1 = tkinter.Canvas(window, width=500, height=500)
        self.canvas2 = tkinter.Canvas(window, width=500, height=500)
        self.canvas1.pack(padx=5, pady=10, side="left")
        self.canvas2.pack(padx=5, pady=10, side="left")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret1, frame1, ret2, frame2 = self.vid1.get_frame

        if ret1 and ret2:
                self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
                self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
                self.canvas1.create_image(0, 0, image=self.photo1, anchor=tkinter.NW)
                self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source1, video_source2):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source1)
        self.vid2 = cv2.VideoCapture(video_source2)

        if not self.vid1.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        ret2 = ""
        if self.vid1.isOpened() and self.vid2.isOpened():
            ret1, frame1 = self.vid1.read()
            ret2, frame2 = self.vid2.read()
            frame1 = cv2.resize(frame1, (500, 500))
            frame2 = cv2.resize(frame2, (500, 500))
            if ret1 and ret2:
                # Return a boolean success flag and the current frame converted to BGR
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            else:
                return ret1, None, ret2, None
        else:
            return ret1, None, ret2, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid1.isOpened():
            self.vid1.release()
        if self.vid2.isOpened():
            self.vid2.release()


def callback():
    global v1,v2
    v1=E1.get()
    v2=E2.get()
    if v1 == "" or v2 == "":
        L3.pack()
        return
    initial.destroy()


v1 = ""
v2 = ""

initial = tkinter.Tk()
initial.title("KEC MEDIA PLAYER")
L0 = tkinter.Label(initial, text="Enter the full path")
L0.pack()
L1 = tkinter.Label(initial, text="Video 1")
L1.pack()
E1 = tkinter.Entry(initial, bd =5)
E1.pack()
L2 = tkinter.Label(initial, text="Video 2")
L2.pack()
E2 = tkinter.Entry(initial, bd =5)
E2.pack()
B = tkinter.Button(initial, text ="Next", command = callback)
B.pack()
L3 = tkinter.Label(initial, text="Enter both the names")

initial.mainloop()


# Create a window and pass it to the Application object
App(tkinter.Tk(),v1, v2)
'''

import tkinter as tk
import PIL.Image
import PIL.ImageTk
import cv2

LARGE_FONT = ("Verdana", 12)

#C:\Users\peted\Documents\Git_Hub\NJIT_REU_Eye_Blinks\blink-detection\000001M_FBN.mp4


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        video_source1 = "000001M_FBN.mp4"
        tk.Tk.__init__(self, *args, **kwargs)

        # open video source
        self.vid1 = DisplayVideo(video_source1)

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(1, weight=1)

        self.frames = {}

        frame = StartPage(container, self)
        frame0 = DisplayVideo(container, self)
        frame1 = DisplayGraph(container, self)

        self.frames[StartPage] = frame
        self.frames[DisplayVideo] = frame0
        self.frames[DisplayGraph] = frame1

        frame.grid(row=0, column=1, sticky="nsew")
        frame0.grid(row=0, column=2, sticky="nsew")
        frame1.grid(row=1, column=1, sticky="nsew")

        self.show_frame(StartPage)
        self.show_frame(DisplayGraph)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

        # Create a canvas that can fit the above video source size
        #self.canvas1 = tk.Canvas(window, width=500, height=500)
        #self.canvas2 = tkinter.Canvas(window, width=500, height=500)
        #self.canvas1.pack(padx=5, pady=10, side="left")
        #self.canvas2.pack(padx=5, pady=60, side="left")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def update(self):
        # Get a frame from the video source
        ret1, frame1, ret2, frame2 = self.vid1.get_frame

        if ret1 and ret2:
            self.photo1 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame1))
           #self.photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame2))
            self.canvas1.create_image(0, 0, image=self.photo1, anchor=tk.NW)
            #self.canvas2.create_image(0, 0, image=self.photo2, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)


class DisplayVideo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)


    #def __init__(self, video_source1, video_source2):
    def __init__(self, video_source1):
        # Open the video source
        self.vid1 = cv2.VideoCapture(video_source1)
        #self.vid2 = cv2.VideoCapture(video_source2)

        if not self.vid1.isOpened():
            raise ValueError("Unable to open video source", video_source1)

    @property
    def get_frame(self):
        ret1 = ""
        #ret2 = ""
        #if self.vid1.isOpened() and self.vid2.isOpened():
        if self.vid1.isOpened():
            ret1, frame1 = self.vid1.read()
            #ret2, frame2 = self.vid2.read()
            frame1 = cv2.resize(frame1, (500, 500))
            #frame2 = cv2.resize(frame2, (500, 500))
            #if ret1 and ret2:
            if ret1:
                # Return a boolean success flag and the current frame converted to BGR
                #return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), ret2, cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                return ret1, cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            else:
                #return ret1, None, ret2, None
                return ret1, None
        else:
            #return ret1, None, ret2, None
            return ret1, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid1.isOpened():
            self.vid1.release()
        #if self.vid2.isOpened():
            #self.vid2.release()


class DisplayGraph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

app = SeaofBTCapp()
app.mainloop()
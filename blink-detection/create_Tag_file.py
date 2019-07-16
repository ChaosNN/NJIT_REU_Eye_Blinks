import os
import pandas as pd

filename = "video.tag"

# Up date the tag file if the tag file exists
def get_tag_data(path):
   data = path.readlines()

   # 1. frame_ID - frame counter, used to obtain time stamp
   # 2. blink_ID - unique blink ID, defined as a sequence of the same blink ID frames
   # 3. non frontal face (NF) - subject is looking sideways and an eye blink occurs, the variable changes from X to N
   # 4. left eye (LE), right eye (RE)
   # 5. eye fully closed (FC) - If the subject's eyes are closed from 90% to 100%, the flag will change from X to C
   # 6. eye not visible (NV) - subjects eye not visible due to physical blockage, bad light conditions, and too quick
   #    movement of the head, will change the variable from X to N
   # 7. face bounding box (F_X, F_Y, F_W, F_H) x and y coordinates, width and height
   # 8. left & right corner positions - RX(right corner x coordinate), LY(left corner y coordinate)
   columns = ["frame_ID", "blink_ID", "NF", "LE_FC", "LE_NV", "RE_FC", "RE_NV", "F_X", "F_Y", "F_W", "F_H", "LE_LX",
              "LE_LY", "LE_RX", "LE_RY", "RE_LX", "RE_LY", "RE_RX", "RE_RY"]
   df_data = pd.DataFrame(data, columns=columns)


def tag_output(path):
    try:
        with open(path, "w") as file:
        file.write("tag files\n")


print(tag_output(filename))

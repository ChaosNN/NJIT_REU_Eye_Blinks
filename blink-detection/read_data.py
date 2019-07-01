import os
import pandas as pd


mypath = 'C:/Users/peted/Downloads/zju/zju'
df = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path'])
pd.set_option('display.max_columns', 4)
for (dirpath, dirnames, filenames) in os.walk(mypath):
    if not filenames:
        print("empty")
    if filenames:
        filenames.append(dirpath)
        df.loc[len(df)] = filenames


print(df.shape[0])

import os
import pandas as pd


'''data_set = 'zju'
mypath = os.path.join(os.getcwd(), 'data_sets\\', data_set)
print('mypath')
print(mypath)
df = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path'])
pd.set_option('display.max_columns', 4)
for (dirpath, dirnames, filenames) in os.walk(mypath):
    if not filenames:
        print("empty")
    if filenames:
        filenames.append(dirpath)
        df.loc[len(df)] = filenames
        #print(filenames)



#print(df)
print(df.shape[0])
print(df.at[10, 'path'])
#print(os.getcwd())

#print(mypath)
'''

df_videodata = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path', 'file_name'])
pd.set_option('display.max_columns', 5)

def read_data(data_set):
    #data_set = 'zju'
    mypath = os.path.join(os.getcwd(), 'data_sets\\', data_set)
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        if not filenames:
            print("empty")
        if filenames:
            print("path")
            print(dirpath)
            filenames.append(dirpath)
            file_name = filenames[0][:-3] + 'png'
            filenames.append(file_name)
            df_videodata.loc[len(df_videodata)] = filenames
    return df_videodata


print(read_data('zju'))
#print(df_videodata['file_name'] = df_videodata['video_file'])
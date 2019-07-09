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

df_videodata = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path', 'file_name', 'folder'])
pd.set_option('display.max_columns', 6)


# gets the information about the file paths of the selected dataset
def read_data(dataset_name):
    mypath = os.path.join(os.getcwd(), 'data_sets\\', dataset_name)
    print(mypath)
    for (dirpath, dirnames, filenames) in os.walk(mypath):
        if not filenames:
            print("empty")
            #print(filenames)
        if filenames:

            filenames.append(dirpath)
            filenames.append(filenames[0][:-3] + 'png')
            dir_split = dirpath.split('\\')
            dir_split = dir_split[len(dir_split) - 1]
            filenames.append(dir_split)

            '''
            # raise ValueError("cannot set a row with "
            # ValueError: cannot set a row with mismatched columns
            try:
                df_videodata.loc[len(df_videodata)] = filenames
            except ValueError:
                #print("does not contain the png graph file")
                #print(filenames)
                df2 = pd.DataFrame(columns=['video_file', 'dat_file', 'text_file', 'path', 'file_name'])
                df2.loc[len(df2)] = filenames
                print("df2")
                print(df2)
                df = df_videodata.append(df2)
                print(df)
            '''

            df_videodata.loc[len(df_videodata)] = filenames
    #result_file = os.path.join(os.getcwd(), 'data_sets\\', dataset_name + '_results', file_name + '.csv')
    #df_videodata.to_csv(result_file)

    return df_videodata


print(read_data('zju'))
#read_data('zju').to_csv()
#print(df_videodata['file_name'] = df_videodata['video_file'])


print(os.path.isdir('C:\\Users\\peted\\Documents\\Git_Hub\\NJIT_REU_Eye_Blinks\\blink-detection\\data_sets\\zju_results'))
dirz = 'C:\\Users\\peted\\Documents\\Git_Hub\\NJIT_REU_Eye_Blinks\\blink-detection\\data_sets\\zju_results'
fileName = 'C:\\Users\\peted\\Documents\\Git_Hub\\NJIT_REU_Eye_Blinks\\blink-detection\\data_sets\\zju_results\\1\\results1.csv'

# checks if directory exists, if not the directory is constructed
def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# checks if the csv file exists, if not the csv file is created
def save_csv(file_path):
    if not os.path.isfile(file_path):
        df_videodata.to_csv(file_path)


print(check_dir(dirz))
dirzz = 'C:\\Users\\peted\\Documents\\Git_Hub\\NJIT_REU_Eye_Blinks\\blink-detection\\data_sets\\zju_results\\1'
print(check_dir(dirzz))
print(save_csv(fileName))
print(df_videodata.iloc[20]['path'])
print('results' + df_videodata.iloc[20]['folder'])

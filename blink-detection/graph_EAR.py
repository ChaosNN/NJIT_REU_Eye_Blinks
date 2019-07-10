# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:00:22 2019

@author: TANMR1
"""

import os
import matplotlib.pyplot as plt
import pandas as pd

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

def graph_EAR_GT(EARs, blink_vals, path, png_filename, folder):
    plt.xlabel('Frame Number')
    plt.ylabel('EAR')
    plt.plot(EARs, 'b')
    plt.plot(blink_vals, 'r')
    data_set = path.split('\\')
    data_set = data_set[len(data_set) - 2] + '_results'
    print(data_set)
    file = os.path.join(os.getcwd(), 'data_sets\\', data_set, folder)
    print(file)
    result = 'results' + folder
    print(os.path.join(file, result + '.csv'))

    try:
        check_dir(file)
        # check_file(file)
        check_file(path + '.csv')
        # os.path.exists(file)

    except IOError:
        print("File exists and will be overwritten")
    finally:
        save_csv(os.path.join(file, result + '.csv'), EARs, blink_vals)
        plt.savefig(os.path.join(file, result + 'graph' + '.png'), bbox_inches='tight')

    plt.close()
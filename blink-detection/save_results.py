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
def save_csv(ear, gt, 
             ga_t, pbp_gat, tp_gat, fp_gat, fn_gat, prec_gat, recall_gat, 
             tcf_t, pbp_tcft, tp_tcft, fp_tcft, fn_tcft, prec_tcft, recall_tcft, 
             h2_t, pbp_h2t, tp_h2t, fp_h2t, fn_h2t, prec_h2t, recall_h2t, 
             h25_t, pbp_h25t, tp_h25t, fp_h25t, fn_h25t, prec_h25t, recall_h25t, 
             h3_t, pbp_h3t, tp_h3t, fp_h3t, fn_h3t, prec_h3t, recall_h3t, 
             h35_t, pbp_h35t, tp_h35t, fp_h35t, fn_h35t, prec_h35t, recall_h35t, 
             file_path, file):
    df = pd.DataFrame(ear, columns=['EAR'])
    
    df['Gen. Avg. Thresh'] = pd.Series(ga_t)    
    df['Predicted Blink Pairs with Gen. Avg. Thresh'] = pd.Series(pbp_gat)
    df['TP with Gen. Avg. Thresh'] = pd.Series(tp_gat)
    df['FP with Gen. Avg. Thresh'] = pd.Series(fp_gat)
    df['FN with Gen. Avg. Thresh'] = pd.Series(fn_gat)
    df['Precision with Gen. Avg. Thresh'] = pd.Series(prec_gat)
    df['Recall with Gen. Avg. Thresh'] = pd.Series(recall_gat)
    
    df['Two Consecutive Frame Thresh '] = pd.Series(tcf_t)    
    df['Predicted Blink Pairs with TCFT'] = pd.Series(pbp_tcft)
    df['TP with TCFT'] = pd.Series(tp_tcft)
    df['FP with TCFT'] = pd.Series(fp_tcft)
    df['FN with TCFT'] = pd.Series(fn_tcft)
    df['Precision with TCFT'] = pd.Series(prec_tcft)
    df['Recall with TCFT'] = pd.Series(recall_tcft)
    
    df['0.2 Thresh'] = pd.Series(h2_t)    
    df['Predicted Blink Pairs with H2T'] = pd.Series(pbp_h2t)
    df['TP with H2T'] = pd.Series(tp_h2t)
    df['FP with H2T'] = pd.Series(fp_h2t)
    df['FN with H2T'] = pd.Series(fn_h2t)
    df['Precision with H2T'] = pd.Series(prec_h2t)
    df['Recall with H2T'] = pd.Series(recall_h2t)

    df['0.25 Thresh'] = pd.Series(h25_t)    
    df['Predicted Blink Pairs with H25T'] = pd.Series(pbp_h25t)
    df['TP with H25T'] = pd.Series(tp_h25t)
    df['FP with H25T'] = pd.Series(fp_h25t)
    df['FN with H25T'] = pd.Series(fn_h25t)
    df['Precision with H25T'] = pd.Series(prec_h25t)
    df['Recall with H25T'] = pd.Series(recall_h25t)

    df['0.3 Thresh'] = pd.Series(h3_t)    
    df['Predicted Blink Pairs with H3T'] = pd.Series(pbp_h3t)
    df['TP with H3T'] = pd.Series(tp_h3t)
    df['FP with H3T'] = pd.Series(fp_h3t)
    df['FN with H3T'] = pd.Series(fn_h3t)
    df['Precision with H3T'] = pd.Series(prec_h3t)
    df['Recall with H3T'] = pd.Series(recall_h3t)

    df['0.35 Thresh'] = pd.Series(h35_t)    
    df['Predicted Blink Pairs with H35T'] = pd.Series(pbp_h35t)
    df['TP with H35T'] = pd.Series(tp_h35t)
    df['FP with H35T'] = pd.Series(fp_h35t)
    df['FN with H35T'] = pd.Series(fn_h35t)
    df['Precision with H35T'] = pd.Series(prec_h35t)
    df['Recall with H35T'] = pd.Series(recall_h35t)    

    df.to_csv(os.path.join(file, file_path + '.csv'))

def check_path(path, folder):
    data_set = path.split('\\')
    data_set = data_set[len(data_set) - 2] + '_results'
    print(data_set)
    file = os.path.join(os.getcwd(), 'data_sets\\', data_set, folder)
    #print(file)
    result = 'results' + folder
    #print(os.path.join(file, result + '.csv'))
    try:
        check_dir(file)
        # check_file(file)
        check_file(path + '.csv')
        # os.path.exists(file)
    except IOError:
        print("File exists and will be overwritten")
    return (result, file)
    
   
def graph_EAR_GT(EARs, gt_vals, png_filename, file_path, file):
    plt.xlabel('Frame Number')
    plt.ylabel('EAR')
    plt.plot(EARs, 'b')
    plt.plot(gt_vals, 'r')
    plt.savefig(os.path.join(file, file_path + 'graph' + '.png'), bbox_inches='tight')
    plt.close()
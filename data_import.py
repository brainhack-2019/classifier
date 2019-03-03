#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:15:13 2019
function to import data

inputs: data_path - path of the data in csv format, including '.csv'
        meta_path - path of the meta file in csv format, inclusing '.csv'
        save_path - path to directory to save output files in
        
outputs:    epoched_data_bi512 - trial length 4 seconds, Ntrials x Nchannels x Nsamples
            epoched_data_bi256 - trial length 2 seconds, Ntrials x Nchannels x Nsamples
            labels - gesture id of each trial
        
   all outputs saved in '.npy' format (numpy array) in the directory specified by save_path
        
@author: shukti02
"""
import numpy as np
import pandas as pd

def data_import(data_path,meta_path,save_path):
    
    all_data = np.genfromtxt(data_path,delimiter=' ',dtype='<f')
    meta_data = pd.read_csv(meta_path)
    labels = meta_data.loc[:]['trials.thisIndex']
    labels = labels.to_numpy();
    
    trig = all_data[:,8]
    trig2 = np.roll(trig,1)
    trig2[0] = 0
    
    trig_diff = trig-trig2
    vec_idx = np.nonzero(trig_diff>15000)[0]
    
    vec_roll = np.roll(vec_idx,1)
    vec_roll[0] = 0
    vec_diff = vec_idx-vec_roll
    vec_idx_new = vec_idx[vec_diff>5]
    
    even = np.array([0,2,4,6])
    odd = np.array([1,3,5,7])
    all_data_bi = all_data[:,odd]-all_data[:,even]
    
    epoched_data_bi512 = np.array([all_data_bi[f:f+512,:] for f in vec_idx_new])
    epoched_data_bi256 = np.array([all_data_bi[f:f+256,:] for f in vec_idx_new])
    epoched_data_bi512 = np.swapaxes(epoched_data_bi512,1,2)
    epoched_data_bi256 = np.swapaxes(epoched_data_bi256,1,2)
    
    np.save(save_path+'/'+'epoched_data_bi512.npy',epoched_data_bi512)
    np.save(save_path+'/'+'epoched_data_bi256.npy',epoched_data_bi256)
    np.save(save_path+'/'+'labels.npy',labels)
    
    return [epoched_data_bi512,epoched_data_bi256,labels]

#----test : uncomment---
#data_dir = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/real_data/sample512'
#data_path = data_dir+'/'+'trial_2_psycho.csv'
#meta_path = data_dir+'/'+'trial_2_psycho_meta.csv'
#save_path = data_dir

#[a,b,c] = data_import(data_path,meta_path,save_path)
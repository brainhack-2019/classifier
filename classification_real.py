#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:57:01 2019

@author: shukti02
"""
import numpy as np
import matplotlib.pyplot as plt
import pyriemann
from sklearn.model_selection import cross_val_score
from os import listdir
from os.path import isfile, join

#data_dir = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/real_data/sample128'
#data_files = [data_dir+'/'+f for f in listdir(data_dir) if '.csv' in f]
#
#all_data = [np.genfromtxt(f,delimiter=',') for f in data_files]
#
#gesture1 = all_data[1]
#gesture1_bi = np.array([gesture1[:,0]-gesture1[:,1], gesture1[:,2]-gesture1[:,3],
#                        gesture1[:,4]-gesture1[:,5], gesture1[:,6]-gesture1[:,7]])
#
#plt.plot(gesture1[:,3])

data_dir = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/real_data/sample512'
data_file = data_dir+'/'+'trial_2_psycho.csv'
all_data = np.genfromtxt(data_file,delimiter=' ',dtype='<f')

#plt.plot(all_data[1:10000,8])

#---------------------to determine the event vector-----------------------
trig = all_data[:,8]
trig2 = np.roll(trig,-1)
trig[trig<0]=0
trig2[trig2<0]=0
trig_diff = trig-trig2
vec_idx = np.nonzero(trig_diff>10000)
vec_roll = np.roll(vec_idx,-1)
vec_diff = vec_idx-vec_roll
#plt.plot(trig_diff[10000:20000])

#vec = trig_diff>10000
#plt.plot(vec)
#
#vec_idx = np.nonzero(trig_diff>10000)
#vec_roll = np.roll(vec_idx,-1)
#vec_diff = vec_idx-vec_roll


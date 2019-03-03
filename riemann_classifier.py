#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:11:15 2019

@author: shukti02
"""


import numpy as np
import pyriemann
from sklearn.model_selection import cross_val_score, cross_val_predict

def riemann_fitting(data,labels):
    
    
    
    cov = pyriemann.estimation.Covariances().fit_transform(data)
    mdm = pyriemann.classification.MDM()
    mdm.fit(cov,labels)
    #pr = mdm.predict(cov)
    accuracy = cross_val_score(mdm, cov, labels, cv=5)
    mean_accuracy = accuracy.mean()
    #prlabels = cross_val_predict(mdm, cov, labels, cv=5)
   
    
    return [mdm,mean_accuracy]

def riemann_predict(mdm,test_data,test_label):
    
    if test_data.ndim == 2:
        a = np.shape(test_data)
        b = (1,)+a
        test_data = np.reshape(test_data,b)
    
    cov = pyriemann.estimation.Covariances().fit_transform(test_data)
    pr_label = mdm.predict(cov)
    
    return pr_label

#-----------------test : works----------------
    
data_dir = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/real_data/sample512'
epoched_data_bi512 = np.load(data_dir+'/'+'epoched_data_bi512.npy')
epoched_data_bi256 = np.load(data_dir+'/'+'epoched_data_bi256.npy')
all_labels = np.load(data_dir+'/'+'all_labels.npy')

X_train = epoched_data_bi256[10:-2,:,:]
Y_train = all_labels[10:-2] 

X_test = epoched_data_bi256[-1,:,:]
Y_test = all_labels[-1:] 


[mdm,mean_accuracy] = riemann_fitting(X_train,Y_train)
pr_labels = riemann_predict(mdm,X_test,Y_test)


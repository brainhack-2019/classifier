# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pyriemann
from sklearn.model_selection import cross_val_score

thumb = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/T_T1.csv'
thumb2 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/T_T2.csv'
thumb3 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/T_T3.csv'
T_trial1 = np.genfromtxt(thumb, delimiter=',')
T_trial2 = np.genfromtxt(thumb2, delimiter=',')
T_trial3 = np.genfromtxt(thumb3, delimiter=',')

hclose = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/HC_1.csv'
hclose2 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/HC_2.csv'
hclose3 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/HC_3.csv'
HC_trial1 = np.genfromtxt(hclose, delimiter=',')
HC_trial2 = np.genfromtxt(hclose2, delimiter=',')
HC_trial3 = np.genfromtxt(hclose3, delimiter=',')

littlef = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/L_L1.csv'
littlef2 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/L_L2.csv'
littlef3 = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/L_L3.csv'
L_trial1 = np.genfromtxt(littlef, delimiter=',')
L_trial2 = np.genfromtxt(littlef2, delimiter=',')
L_trial3 = np.genfromtxt(littlef3, delimiter=',')

X = np.array([np.matrix.transpose(HC_trial1),np.matrix.transpose(HC_trial2),
              np.matrix.transpose(T_trial1),np.matrix.transpose(T_trial2)])
X = X[:,:,0:20000]
Y = np.array([0,0,1,1])

X_test = np.array([np.matrix.transpose(HC_trial3),np.matrix.transpose(T_trial3)])
X_test = X_test[:,:,0:20000]
Y_test = np.array([0,1])

cov = pyriemann.estimation.Covariances().fit_transform(X)
mdm = pyriemann.classification.MDM()
accuracy = cross_val_score(mdm, cov, Y, cv=2)
print(accuracy.mean())

cov_test = pyriemann.estimation.Covariances().fit_transform(X_test)
accuracy_test = cross_val_score(mdm, cov_test, Y_test, cv=2)

#import math
#import time
#import copy
#import pandas as pd
import numpy as np
from os import chdir, getcwd
import csv
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Load training data
# ------------------------------------------------------------------------------

# Set working folder
chdir("/Users/ZatorskiJacek/Git/brainhack")

# Load data into numpy arrays - 1st sample
np_L_L1 = np.genfromtxt('Data/S1-Delsys-15Class/L_L1.csv', delimiter = ",")
np_T_T1 = np.genfromtxt('Data/S1-Delsys-15Class/T_T1.csv',delimiter = ",")                
np_HC_1 = np.genfromtxt('Data/S1-Delsys-15Class/HC_1.csv',delimiter = ",")                
np_M_R1 = np.genfromtxt('Data/S1-Delsys-15Class/M_R1.csv',delimiter = ",")
       
# Load data into numpy arrays - 2nd sample      
np_L_L2 = np.genfromtxt('Data/S1-Delsys-15Class/L_L2.csv', delimiter = ",")
np_T_T2 = np.genfromtxt('Data/S1-Delsys-15Class/T_T2.csv',delimiter = ",")                
np_HC_2 = np.genfromtxt('Data/S1-Delsys-15Class/HC_2.csv',delimiter = ",")                
np_M_R2 = np.genfromtxt('Data/S1-Delsys-15Class/M_R2.csv',delimiter = ",")

# Truncate data to first 5 secs (25% of the total length)
trunc_length = int(np_L_L1.shape[0] / 4)
np_L_L1 = np_L_L1[:trunc_length, :]
np_T_T1 = np_T_T1[:trunc_length, :]
np_HC_1 = np_HC_1[:trunc_length, :]
np_M_R1 = np_M_R1[:trunc_length, :]
np_L_L2 = np_L_L1[:trunc_length, :]
np_T_T2 = np_T_T2[:trunc_length, :]
np_HC_2 = np_HC_2[:trunc_length, :]
np_M_R2 = np_M_R2[:trunc_length, :]


# Concatenates above samples into training data set
L_L_train = np.concatenate((np_L_L1, np_L_L2), axis = 0)
T_T_train = np.concatenate((np_T_T1, np_T_T2), axis = 0)
HC_train = np.concatenate((np_HC_1, np_HC_2), axis = 0)
M_R_train = np.concatenate((np_M_R1, np_M_R2), axis = 0)

# Remove old variables
np_L_L1 = None
np_L_L2 = None
np_T_T1 = None
np_T_T2 = None

np_HC_1 = None
np_HC_2 = None
np_M_R1 = None
np_M_R2 = None

# Concatenate all gestures into one array
train = np.concatenate((L_L_train, T_T_train, HC_train, M_R_train), axis = 0)
train_3d = np.reshape(train, (1, train.shape[0], train.shape[1]))

train_3d = np.concatenate((train_3d, train_3d, train_3d, train_3d), axis = 0)


# Prepare lables
L_L_train_y = np.full(L_L_train.shape, fill_value = 0, dtype=None, order='C')
T_T_train_y = np.full(T_T_train.shape, fill_value = 1, dtype=None, order='C')
HC_train_y = np.full(HC_train.shape, fill_value = 2, dtype=None, order='C')
M_R_train_y = np.full(M_R_train.shape, fill_value = 3, dtype=None, order='C')


# Concatenate all gesture labels into one array
train_y = np.concatenate((L_L_train_y, T_T_train_y, HC_train_y, M_R_train_y), axis = 0)
train_y_3d = np.reshape(train_y, (1, train_y.shape[0], train_y.shape[1]))

train_y_3d = np.concatenate((train_y_3d, train_y_3d, train_y_3d, train_y_3d), axis = 0)


# Remove old variables
L_L_train = None
T_T_train = None
HC_train = None
M_R_train = None

L_L_train_y = None
L_L_train_y = None
HC_train_y = None
M_R_train_y = None



# ------------------------------------------------------------------------------
# Load testing data
# ------------------------------------------------------------------------------

if False:
    # Load data into numpy arrays - 3rd sample
    L_L_test = np.genfromtxt('Data/S1-Delsys-15Class/L_L3.csv', delimiter = ",")
    T_T_test = np.genfromtxt('Data/S1-Delsys-15Class/T_T3.csv',delimiter = ",")                
    HC_test = np.genfromtxt('Data/S1-Delsys-15Class/HC_3.csv',delimiter = ",")                
    M_R_test = np.genfromtxt('Data/S1-Delsys-15Class/M_R3.csv',delimiter = ",")
    
    # Truncate data to first 5 secs (25% of the total length)
    L_L_test = L_L_test[:trunc_length, :]
    T_T_test = T_T_test[:trunc_length, :]
    HC_test = HC_test[:trunc_length, :]
    M_R_test = M_R_test[:trunc_length, :]


# ------------------------------------------------------------------------------
# Explore the data
# ------------------------------------------------------------------------------

if False:
    plt.plot(L_L_train[:, 1])
    plt.ylabel('L_L_channel_1')
    plt.show()
    
    plt.plot(L_L_train[:, 2])
    plt.ylabel('L_L_channel_2')
    plt.show()



# ------------------------------------------------------------------------------
# Correlations
# ------------------------------------------------------------------------------

corr_L_L = np.corrcoef(L_L_train.T)
corr_T_T = np.corrcoef(T_T_train.T)
corr_HC = np.corrcoef(HC_train.T)
corr_M_R = np.corrcoef(M_R_train.T)

def matrix_norm(matrix):
    
    return(np.sqrt(np.nansum(np.square(matrix))))
    
    
corr_L_L_test = np.corrcoef(L_L_test.T)


matrix_norm(corr_L_L_test - corr_L_L)
matrix_norm(corr_L_L_test - corr_T_T)
matrix_norm(corr_L_L_test - corr_HC)
matrix_norm(corr_L_L_test - corr_M_R)



# ------------------------------------------------------------------------------
# Load data into numpy arrays - 1st sample
trial_2_psycho = np.genfromtxt('Data/brainhack/trial_2_psycho.csv',\
                               delimiter = " ",\
                               dtype='<f')

plt.plot(trial_2_psycho[:20000, 8])
plt.ylabel('')
plt.show()

aux32 = trial_2_psycho[:, 8]
aux32_lag = np.roll(aux32, -1)

plt.plot(aux32[:20000])
plt.ylabel('')
plt.show()


# Peak detection
PEAK_THRESHOLD = 15000
thresh = (aux32 > PEAK_THRESHOLD).astype('uint8')
plt.plot(thresh[:5000])
plt.ylabel('')
plt.show()


# signal - lag
aux32_diff = aux32 - aux32_lag
plt.plot(aux32_diff[:20000])
plt.ylabel('')
plt.show()


# ------------------------------------------------------------------------------
# Training pyriemann + svm
# https://github.com/alexandrebarachant/pyRiemann
# ------------------------------------------------------------------------------

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# load your data
# X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
# y = ... # the labels

# Prepare the data
# X = train_3d
# y = train_y_3d


# build your pipeline
covest = Covariances()
ts = TangentSpace()
svc = SVC(kernel='linear')

clf = make_pipeline(covest,ts,svc)
# cross validation
accuracy = cross_val_score(clf, train_3d, train_y_3d)

print(accuracy.mean())


# ------------------------------------------------------------------------------
# Training pyriemann -- crashes silently
# https://github.com/alexandrebarachant/pyRiemann
# ------------------------------------------------------------------------------

import pyriemann
from sklearn.model_selection import cross_val_score

# load your data
# X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
# y = ... # the labels

# Prepare the data
X = train_3d
y = train_y_3d


# estimate covariances matrices
cov = pyriemann.estimation.Covariances().fit_transform(X)

# cross validation
mdm = pyriemann.classification.MDM()

accuracy = cross_val_score(mdm, cov, y)

print(accuracy.mean())











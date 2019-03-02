import math
import time
import copy
import pandas as pd
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
HC_train = np.concatenate((np_HC_1, np_HC_1), axis = 0)
M_R_train = np.concatenate((np_M_R1, np_M_R2), axis = 0)


# ------------------------------------------------------------------------------
# Load testing data
# ------------------------------------------------------------------------------

# Load data into numpy arrays - 1st sample
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

plt.plot(L_L_train[:, 1])
plt.ylabel('L_L_channel_1')
plt.show()















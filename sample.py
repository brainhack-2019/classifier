# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt

data_file = '/media/shukti02/BluePearl/BrainHack_Warsaw/Data/sample_data/S1-Delsys-15Class/T_T1.csv'
my_data = np.genfromtxt(data_file, delimiter=',')
plt.figure(1)
plt.plot(my_data[1:20000,0])
plt.figure(2)
plt.plot(my_data[1:20000,1])
plt.figure(3)
plt.plot(my_data[1:20000,2])
plt.figure(4)
plt.plot(my_data[1:20000,3])
plt.figure(5)
plt.plot(my_data[1:20000,4])
plt.figure(6)
plt.plot(my_data[1:20000,5])
plt.figure(7)
plt.plot(my_data[1:20000,6])
plt.figure(8)
plt.plot(my_data[1:20000,7])
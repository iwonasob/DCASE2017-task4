# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 14:27:41 2017

@author: sobier
"""

import matplotlib.pyplot as plt
import pickle
import config as cf

plt.ion()
plt.close('all')

[results, timestamps] = pickle.load(open("/home/sobier/DCASE2017/task4_Iwona/models/crnn_24timestamps.pkl",'rb'))

for idx, h in enumerate(results[40:50]):
    plt.figure()
    plt.plot(timestamps[idx][0])
    plt.title(','.join(h))

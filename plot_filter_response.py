# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:37:20 2017

@author: sobier
"""
import matplotlib.pyplot as plt
import pickle
import config as cf
import numpy as np

plt.ion()

layer_name = 'conv2d_1'
images = pickle.load(open(cf.model_path+"filter_response.pkl"+str(layer_name),'rb'))
#layer_name = 'convolution2d_14'
#images = pickle.load(open("/home/abr/xchange/sobier/pipeline_final/plots/filter_response_"+str(layer_name)+".pkl",'rb'))


#####
plt.close('all')

for idx,f in enumerate(images):
    plt.figure()
#    plt.imshow(np.abs(f[0,:,:]+f[1,:,:]+f[2,:,:]),aspect='auto')
    plt.imshow(np.abs(f),aspect='auto')
    plt.colorbar()
    plt.title("Response to the filter "+str(idx))
    plt.show()

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:05:19 2017

@author: sobier
"""
import sys
sys.path.insert(0, '/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/evaluation')
import os
import numpy as np
import config as cf
from feature_extraction import *
from keras.models import *
from evaluation import *
import re
import csv
from vis.visualization import visualize_saliency


if __name__ == '__main__':
    model_path=cf.model_path
    dir_results= cf.dir_results
    test_dataset=cf.test_dataset

    ev = Task4Evaluation(model_path,
                         dir_results,
                         test_dataset)
                         
    x_test=ev.reshape_data(ev.test_npy_list)
    model=load_model(cf.model_path+'.h5')
    layer_name = "dense_1"
    layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]
    heatmaps=[]
    pred_class=[]
    for x in xrange(len(x_test)-1):
        prediction=model.predict(x_test[x:x+1])
        y_pred = prediction >0.5
        if np.sum(y_pred)==0:
            y_pred[0,np.argmax(prediction)]=1
        heatmap=[]
        for y in np.nonzero(y_pred)[1]:
            print(y)
            h= visualize_saliency(model, layer_idx, y, x_test[x:x+1])
            heatmap.append(h)
        pred_class.append(y_pred)
        
    with open(cf.model_path+"heatmaps_multiclass.pkl", "wb") as f:
        pickle.dump([pred_class, heatmaps], f, protocol=pickle.HIGHEST_PROTOCOL)
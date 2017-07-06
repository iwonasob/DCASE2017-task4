# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:40:21 2017

__author__ = 'Iwona Sobieraj'
"""
import sys
sys.path.insert(0, '/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/evaluation')
import os
import numpy as np
import config as cf
from feature_extraction import *
from keras.models import *
import re
import csv
from TaskAEvaluate import *
from vis.visualization import visualize_saliency
#from IPython.core.debugger import Tracer

class Task4Evaluation:
    def __init__(self,
                 model_path,
                 dir_results,
                 test_dataset):
        
        self.model_path=model_path  
        self.dir_results = dir_results
        self.test_dataset = test_dataset
        if not os.path.isdir(self.dir_results):
            os.makedirs(self.dir_results)
            
        print('Load dataset from file %s ... ' % self.test_dataset)
        data=np.load(self.test_dataset) 
        self.test_fn_wav_list = data['file_name']
        self.test_npy_list= data['feature_arrays']
        self.test_label_vec=data['label_vectors']
        self.saliency_layer_name = "dense_1"
        
        
    def run(self):
    # initialize model
        print('Load model')
        model = load_model(self.model_path+'.h5')
        self.layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name ==  self.saliency_layer_name][0]
        x_test=self.reshape_data(self.test_npy_list) 
        print('Predict tags and timestamps')
        self.predict(model, x_test) 
 
     
    def reshape_data(self, data_list):
        X=np.array(data_list)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return X
        
    def predict(self, model, X):
        prediction = model.predict(X, batch_size=10, verbose=1)
        y_pred = prediction >0.5
       
        for idx, y in enumerate(y_pred):
            if sum(y)==0:
                y_pred[idx][np.argmax(prediction[idx])]=1   

        timestamps=[] 
        results=[]
        dataset=DatasetCreator()
        for idx, fn_wav in enumerate(self.test_fn_wav_list):
            label=dataset.vec2label(y_pred[idx]) #write as label not code!
            print("Detected " + str(','.join(label))+ " in " + str(idx))
            fn= os.path.basename(fn_wav[0])[1:]
            [start, end]=re.findall('[0-9]+\.[0-9]+', fn)
            results.append((fn, start, end, ','.join(label)))
            timestamp=[]
            for y in np.nonzero(y_pred[idx]):   
                print(y)
                h = visualize_saliency(model, self.layer_idx, y, X[idx:idx+1])
                h_time=np.sum(np.sum(h,axis=2),axis=0)
                h_time_threshold= h_time>np.max(h_time)*0.5
                timestamp.append(h_time_threshold)  
            timestamps.append(timestamp) 
            
        print('Finished extracting saliency') 
        
        with open(cf.model_path+"timestamps.pkl", "wb") as f:
            pickle.dump([results, timestamps], f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print('Writing results to a file '+cf.save_filename)    
        with open(cf.save_filename, 'wt') as f:
            writer = csv.writer(f, delimiter='\t',quoting = csv.QUOTE_NONE, quotechar='') 
            for result in results:
                writer.writerow(result)
        f.close()
        
        print('Writing results to a file '+cf.eval_filename)
        evaluateMetrics(cf.test_weak_label, cf.save_filename, cf.eval_filename)
        

        
if __name__ == '__main__':
    
    model_path=cf.model_path
    dir_results= cf.dir_results
    test_dataset=cf.test_dataset
       
    ev = Task4Evaluation(model_path,
                         dir_results,
                         test_dataset)
                    
    ev.run() 

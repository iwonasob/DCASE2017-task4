# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:14:27 2017

__author__ = 'Iwona Sobieraj'
"""
#from IPython.core.debugger import Tracer

import sys
import os
import csv
import numpy as np
import glob
import time
import cPickle as pickle
from multiprocessing import *
import librosa
from sklearn.preprocessing import * 
import tqdm  
import config as cf 

do_parallel=False 

def feature_extraction(feature, fn_wav, dir_target_features):
    """ Extract features from audio files and save them to npy arrays
    Args:
        feature (string):    Type of the features
        fn_wav (string):     Path to the wav file
        dir_target_features: Path to the target directory 
    """
    x, fs = librosa.load(fn_wav, sr=cf.sr, mono=True)
   
    if feature == "stft":
        X = np.abs(librosa.stft(x, n_fft=cf.n_fft , hop_length=cf.hop_length, win_length=cf.win_length, window=cf.win, center=True))
    elif feature == "mel":
        X = librosa.feature.melspectrogram(x, sr=fs, n_mels=cf.n_mels, n_fft=cf.n_fft, hop_length=cf.hop_length, power=2)
    elif feature == "logmel":
        X = np.log(1+librosa.feature.melspectrogram(x, sr=fs, n_mels=cf.n_mels, n_fft=cf.n_fft, hop_length=cf.hop_length, power=2))
    else:
        raise ValueError('Invalid feature type!')
        
    fn_npy = os.path.join(dir_target_features, os.path.basename(fn_wav).replace('.wav', '_'+feature ))
    assert np.all(np.logical_not(np.isnan(X)))

    np.save(fn_npy, X)

   
def choose_scaler(datasetCreator, feature_list):
    """ Trains a scaler for the data or loads an existing one for testing data
    Args:
        feature_list (list):   List of feature arrays
    Returns:
        
    """
    n = len(feature_list)  # number of rows
    batch_size = 1000  # number of rows in each call to partial_fit
    index = 0  # helper-var
    
    if datasetCreator.mode == "training":
        print("Training the scaler")
        scaler = StandardScaler()    
        while index < n:
            partial_size = min(batch_size, n - index)  
            partial_x = np.hstack(feature_list[index:index+partial_size])
            scaler.partial_fit(partial_x.transpose())
            index += partial_size
        pickle.dump(scaler, open( datasetCreator.dir_model+datasetCreator.feature+'_scaler.p', 'wb'))
        
    elif datasetCreator.mode == "testing":
        scaler = pickle.load(open(datasetCreator.dir_mocalerdel+datasetCreator.feature+'_scaler.p', 'rb'))
    return scaler
    
    
def apply_normalizer(datasetCreator, fn_npy):
    """ Trains a scaler for the data or loads an existing one for testing data
    Args:
        feature_list (list):   List of feature arrays
    Returns:
        
    """
    X=np.load(fn_npy)
    X= normalize(X, norm='l2')
    assert np.all(np.logical_not(np.isnan(X)))
    np.save(fn_npy,X)
    
def apply_scaler(datasetCreator, scaler, fn_npy):
    """ Trains a scaler for the data or loads an existing one for testing data
    Args:
        feature_list (list):   List of feature arrays
    Returns:
        
    """
    X=np.load(fn_npy)
    X=scaler.transform(X.transpose())
    assert np.all(np.logical_not(np.isnan(X)))
    np.save(fn_npy,X.transpose())
    
def apply_padding(datasetCreator, fn_npy):
    X=np.load(fn_npy)
    if X.shape[1] < datasetCreator.max_length:
        X=np.hstack((X,np.zeros((X.shape[0],(datasetCreator.max_length-X.shape[1])))))
    elif X.shape[1] > datasetCreator.max_length:
        X=X[:,:datasetCreator.max_length]        
    assert np.all(np.logical_not(np.isnan(X)))
    np.save(fn_npy,X)   
    
class DatasetCreator:
    def __init__(self,
                 source_csv=cf.source_csv,
                 dir_source_wav=cf.dir_source_wav,
                 dir_target=cf.dir_target,
                 dir_model=cf.dir_model,
                 feature=cf.feature,
                 mode=cf.mode,
                 max_length=cf.max_length):
        """ Initialize class
        Args:
            dir_source_txt (string): Source directory with TXT files containing class-wise file-lists
            dir_source_wav (string): Core directory with WAV files
            dir_target (string): Target directory for dataset
        """
        self.source_csv = source_csv
        self.label_list = cf.label_list
        self.dir_source_wav = dir_source_wav
        self.dir_target = dir_target
        self.dir_model = dir_model
        self.fn_wav_list = DatasetCreator.load_file_lists(source_csv, dir_source_wav)
        self.feature=feature
        self.mode=mode
        self.max_length=max_length
        self.full_dataset=self.dir_target+"full_set_"+self.feature+".p"
        with open(self.label_list) as f:
            self.no_classes = sum(1 for _ in f)
        if not os.path.isdir(self.dir_target):
            os.makedirs(self.dir_target)
        if not os.path.isdir(self.dir_model):
            os.makedirs(self.dir_model)
            
    def run(self,
            extract_features,
            normalize_features,
            scale_features,
            normalize_length,
            aggregate_features):
        
        if extract_features:
            print('Remove existing features')
            files = glob.glob(os.path.join(self.dir_target, '*.npy'))
            for _file in files:

                os.remove(_file)
            if do_parallel:
                pool = Pool()
                for fn_wav, label in tqdm.tqdm(self.fn_wav_list.items()):
#                    if i % 100:
#                        print('Load feature file %d/%d' % (i + 1, num_files))
                    pool.apply_async(feature_extraction, args=(self.feature, fn_wav, self.dir_target))
                pool.close()
                pool.join()
            else:
                for fn_wav, value in tqdm.tqdm(self.fn_wav_list.items()):
                    feature_extraction(self.feature, fn_wav, self.dir_target)
            
        if normalize_features:
            print('Normalizing features with L2 norm')
            if do_parallel:
                pool = Pool()
                for fn_wav, label in tqdm.tqdm(self.fn_wav_list.items()):
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    pool.apply_async(apply_normalizer, args=(self, fn_npy))                    
                pool.close()
                pool.join()
            else:
                for fn_wav, value in self.fn_wav_list.items():
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    apply_normalizer(self, fn_npy)
            
        if scale_features:
            print('Scaling features to zero mean and unit variance')
            fn_total=[]
            for fn_wav, label in self.fn_wav_list.items():
                fn_npyTrue = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                fn_total.append(np.load(fn_npy))
            scaler=choose_scaler(self, fn_total)
            if do_parallel:
                pool = Pool()
                for fn_wav, label in tqdm.tqdm(self.fn_calerwav_list.items()):
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    pool.apply_async(apply_scaler, args=(self, scaler, fn_npy))                    
                pool.close()
                pool.join()
            else:
                for fn_wav, value in self.fn_wav_list.items():
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    apply_scaler(self, scaler, fn_npy)
                    
        if normalize_length:
            print("Normalizing length of the files")
            if do_parallel:
                pool = Pool()
                for fn_wav, label in tqdm.tqdm(self.fn_wav_list.items()):
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    pool.apply_async(apply_padding, args=(self, fn_npy))                    
                pool.close()
                pool.join()
            else:
                for fn_wav, value in self.fn_wav_list.items():
                    fn_npy = os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                    apply_padding(self, fn_npy)
            
                
        if aggregate_features:
            print("Creating a dataset file")
            npy_list=[]
            vec_label=[]
            for fn_wav, label in self.fn_wav_list.items():
                fn_npy_path=os.path.join(self.dir_target, os.path.basename(fn_wav).replace('.wav', '_'+feature+'.npy'))
                if os.path.exists(fn_npy_path):
                    npy=np.load(fn_npy_path)
                    npy_list.append(npy)
                    vec_label.append(self.label2vec(label))
                
            dataset = {}  
            dataset['file_name'],  dataset['feature_arrays'], dataset['label_vectors']= self.fn_wav_list.items(), npy_list, vec_label
            pickle.dump( dataset, open( self.full_dataset, 'wb' ))
            print("Da10taset saved: "+self.full_dataset )
    
    @staticmethod        
    def load_file_lists(source_csv, dir_source_wav):
        """ Load class-wise file lists from SCV files
        Args:
            source_csv (string): Path to CSV file with file-lists
            dir_source_core (string): Location of WAV fcalercaleriles
        Returns:
            fn_wav_list: Dictionary with key: file path and value: label            
        """
        fn_wav_list = {}
        f = open(source_csv, 'rt')
        try:
            reader = csv.reader(f)
            for row in reader:
                file_id=row[0]
                start_timestamp=row[1]
                end_timestamp=row[2]
                label=row[4]
                filename= "Y"+file_id+"_"+start_timestamp+"_"+end_timestamp+".wav"
                filepath=os.path.join(dir_source_wav, filename)
                if os.path.exists(filepath):
                    fn_wav_list[filepath]=label
        finally:
            f.close()        
        return fn_wav_list
    
    def label2vec(dataset, label):
        """ Convert labels into binary vectors
        Args:
            file_path (string): Path to a file with file-lists
        Returns:
            vec: Vector with encoded list of labels            
        """
        vec=np.zeros(dataset.no_classes)
        with open(dataset.label_list) as f:
            content = f.readlines()
        label_list = label.split(',')
        for l in label_list:
            index = [x for x in range(len(content)) if l in content[x].lower()]
            vec[index]=1    
        f.close()
        return vec

    def vec2label(dataset, vector):
        """ Convert vectors into list of labels
        Args:
            file_path (string): Path to a file with file-lists
        Returns:
            vec: Vector with encoded list of labels            
        """
        with open(dataset.label_list, 'rb') as f:
            reader = csv.reader(f,  delimiter='\t', )
            content = list(reader)
        label_list=[]
#        Tracer()()
        for idx in np.nonzero(vector)[0][:]:
            label_list.append(content[idx][1])
        f.close()
        return label_list          
            
if __name__ == '__main__':
    extract_features = False
    normalize_features = False
    scale_features = False 
    normalize_length = False
    aggregate_features = True
    
    mode = cf.mode
    feature= cf.feature 
    max_length=cf.max_length 
    
    source_csv=cf.source_csv
    label_list=cf.label_list
    dir_source_wav=cf.dir_source_wav
    dir_target=cf.dir_target
    dir_model=cf.dir_model
    
    
    dataset = DatasetCreator(source_csv,
                                    dir_source_wav,
                                    dir_target,
                                    dir_model,
                                    feature,
                                    mode,
                                    max_length
                                    )
                                    
    dataset.run(extract_features=extract_features,
                scale_features=scale_features,
                normalize_features = normalize_features,
                normalize_length = normalize_length,
                aggregate_features=aggregate_features)
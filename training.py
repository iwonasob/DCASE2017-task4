# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:40:21 2017

__author__ = 'Iwona Sobieraj'
"""
from IPython.core.debugger import Tracer

import os
import cPickle as pickle
import time
import numpy as np
import librosa
import config as cf
from keras.optimizers import adam
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, GRU, Reshape
from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, add
from keras.layers.merge import Add
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from feature_extraction import *
from keras import regularizers

class Task4Training:
    def __init__(self,
                 model_path,
                 train_dataset,
                 test_dataset,
                 model_num_epochs=200,
                 model_batch_size=128): 
                     
        self.model_path=model_path             
        self.model_num_epochs = model_num_epochs
        self.model_batch_size = model_batch_size   
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
#        if not os.path.isdir(self.dir_results):
#            os.makedirs(self.dir_results)
        self.optimizer = adam(lr=.001)
        self.load_dataset()
        self.num_classes=self.test_label_vec[0].shape[0]
        self.sample_length=self.test_npy_list[0].shape[1]

    def load_dataset(self):
    # correct this one, remove double code!
        print('Load dataset from file %s ... ' % self.train_dataset)
        t = time.time()
        data= np.load(self.train_dataset)
        self.train_fn_wav_list = data['file_name']
        self.train_npy_list= data['feature_arrays']
        self.train_label_vec=data['label_vectors']
        
        print('Load dataset from file %s ... ' % self.test_dataset)
        data=np.load(self.test_dataset) 
        self.test_fn_wav_list = data['file_name']
        self.test_npy_list= data['feature_arrays']
        self.test_label_vec=data['label_vectors']
        print('    ... in %2.2f seconds' % (time.time() - t))     
         
         
    def run(self):
    # initialize model
        print('Initialize new model')
#        model = self.prepare_model_cbrnn()
        model = self.prepare_model_crnn()
#        model = self.prepare_model_baseline()
        print('Compile new model')
        model = self.compile_model(model)
        
        tmp_path=cf.model_tmp_path+"weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(tmp_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')
        callbacks_list = [checkpoint, earlystopping]        
        
        print('Shingle')
        if cf.shingle:
            for idx, t in enumerate(self.train_npy_list):
                self.train_npy_list[idx]=librosa.feature.stack_memory(t, n_steps=4)

        x_train=self.reshape_data(self.train_npy_list)  
        y_train=np.array(self.train_label_vec)
        x_test=self.reshape_data(self.test_npy_list)
        y_test=np.array(self.test_label_vec)   
        
#        print('Reshape the train data')
#        x_train,y_train=self.reshape_data_mlp(self.train_npy_list,self.train_label_vec)
#        print('Reshape the test data')
#        x_test,y_test=self.reshape_data_mlp(self.test_npy_list,self.test_label_vec)
        
#        model.load_weights('/home/sobier/DCASE2017/task4_Iwona/models/tmp/crnn_24(5, 5)_weights-improvement-41-0.94.hdf5')
        
        print('Fit new model')
        hist = model.fit(x_train,
                  y_train,
                  nb_epoch=self.model_num_epochs,
                  batch_size=self.model_batch_size,
                 # shuffle=self.model_shuffle,
                  # callbacks=[history, early, checkpoint],
                  callbacks=callbacks_list, 
                  validation_split=0.2,
                 #validation_data=(feat_validation, target_validation),
                 #class_weight=dict(zip(classes, class_weight_vec))
                  )
        print(hist.history)
        model.save(self.model_path+'.h5')
        score = model.evaluate(x_test, y_test, verbose=1)  
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])#    from keras.layers import merge, Convolution2D, Input
        return model

    def reshape_data(self, data_list):
        X=np.array(data_list)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return X
        
        
    def reshape_data_mlp(self, X_list, Y_list):
        X=np.array(X_list)
        Y_reshaped= [np.tile(y,( X.shape[2], 1)).transpose() for y in Y_list]
        Y=np.array(Y_reshaped)
        X = np.hstack(X).transpose()
        Y = np.hstack(Y).transpose()
        return X,Y
        
    def prepare_model_crnn(self,
                            input_shape=(40,862,1),
                            dropout = .25):
                
        n_filters=cf.n_filters 
        kernel_size=cf.kernel_size               
        model = Sequential()
    
        #add dropout on input layer ??
        model.add(Conv2D(n_filters, kernel_size=kernel_size,activation='relu', padding="same",input_shape=input_shape))    
        model.add(MaxPooling2D(pool_size=(5, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)) 
        
        model.add(Dropout(dropout))
        model.add(Conv2D(n_filters, kernel_size, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)) 
        
        model.add(Dropout(dropout))
        model.add(Conv2D(n_filters, kernel_size, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
        
        model.add(Dropout(dropout))        
        model.add(Conv2D(n_filters, kernel_size, activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
        
        model.add(Reshape((self.sample_length,n_filters)))
        
        model.add(GRU(n_filters,return_sequences=True))
        model.add(GRU(n_filters,return_sequences=True))
        
        model.add(MaxPooling1D(pool_size=input_shape[1]))
        model.add(Flatten())
        
        model.add(Dense(self.num_classes, activation='sigmoid', activity_regularizer=regularizers.l1(0.01)))
#        kernel_regularizer=regularizers.l1(0.01)
    #
        return model
        
        
    def prepare_model_cbrnn(self,
                            input_shape=(40,862,1),
                            dropout = .25):
                
        n_filters=cf.n_filters                
        model = Sequential()
    
        #add dropout on input layer ??
        model.add(Conv2D(n_filters, kernel_size=(5, 5),activation='relu', padding="same",input_shape=input_shape))    
        model.add(MaxPooling2D(pool_size=(5, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)) 
        
        model.add(Dropout(dropout))
        model.add(Conv2D(n_filters, (5, 5), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)) 
        
        model.add(Dropout(dropout))
        model.add(Conv2D(n_filters, (5, 5), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
        
        model.add(Dropout(dropout))        
        model.add(Conv2D(n_filters, (5, 5), activation='relu', padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(BatchNormalization(axis=3, momentum=0.99, epsilon=0.001))
        
        model.add(Reshape((self.sample_length,n_filters)))
        model.add(Bidirectional(GRU(n_filters,return_sequences=True)))
        model.add(Bidirectional(GRU(n_filters,return_sequences=True)))
        
        model.add(MaxPooling1D(pool_size=input_shape[1]))
        model.add(Flatten())
        
        model.add(Dense(self.num_classes, activation='sigmoid'))
    #
        return model
    def prepare_model_crnn_res(self,
                            input_shape=(40,862,1),
                            dropout = .25):


        input_spectrogram = Input(shape=input_shape)

        #add dropout on input layer ??
        y_1= Conv2D(96, kernel_size=(5, 5),activation='relu', padding="same")(input_spectrogram)
        y_1=MaxPooling2D(pool_size=(5, 1))(y_1)
        y_1=BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(y_1) 
        
        y_2=Dropout(dropout)(y_1)
        y_2=Conv2D(96, (5, 5), activation='relu', padding="same")(y_2)
        y_2=MaxPooling2D(pool_size=(2, 1))(y_2)
        y_2=BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(y_2) 
        
        y_3=Dropout(dropout)(y_2)
        y_3=Conv2D(96, (5, 5), activation='relu', padding="same")(y_3)
        y_3=MaxPooling2D(pool_size=(2, 1))(y_3)
        y_3=BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(y_3)
        
        y_4=Dropout(dropout)(y_3)        
        y_4=Conv2D(96, (5, 5), activation='relu', padding="same")(y_4)
        y_4=MaxPooling2D(pool_size=(2, 1))(y_4)
        y_4=BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(y_4)
        y_4=Reshape((self.sample_length,96))(y_4)
        
        y_5=GRU(96,return_sequences=True)(y_4)
        y_6 = add([y_4, y_5])
                
        y_7=GRU(96,return_sequences=True)(y_6)
        y_8 = add([y_6, y_7])

        y_8=Flatten()(y_8)
        predictions=Dense(self.num_classes, activation='sigmoid')(y_8)

        model = Model(input=input_spectrogram, output=predictions)
        
        return model
        
        
    def prepare_model_baseline(self,
                               input_shape=(160),
                               dropout = .2):
                                
        model = Sequential()
        model.add(Dense(50, activation='relu', input_dim=input_shape))
        
        model.add(Dense(50, activation='relu'))
    
        model.add(Dense(self.num_classes, activation='sigmoid'))
    #
        return model

    def compile_model(self, model):
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        return model


if __name__ == '__main__':
    
    train_dataset=cf.train_dataset
    test_dataset=cf.test_dataset
    model_path=cf.model_path
       
    tr = Task4Training(model_path,
                       train_dataset,
                       test_dataset)
                    
    tr.run() 

# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:14:27 2017

Contains all the parameters of the system

__author__ = 'Iwona Sobieraj'
"""

mode = "training"

feature="logmel"
max_length=862 # convert to (fs/fft_bins)*(1/hop)

# feature parameters
sr = 44100
n_fft = 1024
hop_length = 512
win_length = 1024
win = 'hann'
n_mel = 40
n_filters = 24
kernel_size=(20,5)

model_name="crnn_"+str(n_filters)+str(kernel_size)
#model_name="crnn_"+str(n_filters)+str(kernel_size)+"_activationl1"
shingle=False

# paths
source_csv="/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/"+mode+"_set_total.csv"
label_list="/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/label_list.txt"
#label_list="/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/sound_event_list_17_classes.txt"
dir_source_wav="/mnt/IDMT-WORKSPACE/DATA-STORE/sobier/dcase2017/task4/"+mode+"_set_"+mode+"_set_audio_formatted_and_segmented_downloads/"
dir_target="/mnt/IDMT-WORKSPACE/DATA-STORE/sobier/dcase2017/task4/"+mode+"_set_features/"
dir_model="/home/sobier/DCASE2017/task4_Iwona/models/"
dir_results="/home/sobier/DCASE2017/task4_Iwona/results/"
train_dataset="/home/avdata/audio/own/Dcase2017/task4/train/full_set_logmel.p"
test_dataset="/home/avdata/audio/own/Dcase2017/task4/test/full_set_logmel.p"
#test_dataset="/home/sobier/DCASE2017/task4_Iwona/full_set_logmel.p"
#train_dataset=test_dataset
test_weak_label="/home/sobier/DCASE2017/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars/groundtruth_weak_label_testing_set.csv"

   
model_path=dir_model+model_name   
model_tmp_path=dir_model+"tmp/"+model_name+"_"

save_filename=dir_results+model_name+"_results.txt"
eval_filename=dir_results+model_name+"_eval.txt"

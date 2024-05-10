#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:06:27 2024
1. Follow tensor flow sequential model
2. Follow https://github.com/talhaanwarch/youtube-tutorials 
@author: tzcheng
"""
import os
import mne
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv1D,BatchNormalization,LeakyReLU,MaxPool1D,GlobalAveragePooling1D,Dense,Dropout,AveragePooling1D,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session

#%%####################################### Load the data
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times

did_pca = '_ffr' # without or with pca "_pcffr"
filename_ffr_ba = 'group_ba' + did_pca
filename_ffr_mba = 'group_mba' + did_pca
filename_ffr_pa = 'group_pa' + did_pca

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

baby_or_adult = 'cbsA_meeg_analysis' # baby or adult
input_data = 'sensor' # ROI or wholebrain or sensor or pcffr
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

if input_data == 'sensor':
    ffr_ba = np.load(root_path + baby_or_adult + '/MEG/FFR/' + filename_ffr_ba + '_sensor.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_sensor.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_sensor.npy',allow_pickle=True)
elif input_data == 'ROI':
    ffr_ba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_ba + '_morph_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_morph_roi.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_morph_roi.npy',allow_pickle=True)
elif input_data == 'wholebrain':
    ffr_ba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_ba + '_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_morph.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_morph.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

all_score = []
X = np.concatenate((ffr_ba,ffr_mba,ffr_pa),axis=0)
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_ba)),np.repeat(2,len(ffr_ba)))) #0 is for mmr1 and 1 is for mmr2

rand_ind = np.arange(0,len(X))
random.Random(0).shuffle(rand_ind)
X = X[rand_ind,0,:]
y = y[rand_ind]

#%%####################################### 1D NN
## preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
scaler = StandardScaler()
X_train_reshape = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_reshape = scaler.fit_transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")


## build the model
def cnnmodel():
    clear_session()
    model=Sequential()
    model.add(Conv1D(filters=5,kernel_size=3,strides=1,input_shape=(1101,3)))#1
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2,strides=2))#2
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))#3
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2,strides=2))#4
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))#5
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))#6
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))#7
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))#8
    model.add(Conv1D(filters=5,kernel_size=3,strides=1))#9
    model.add(LeakyReLU())
    model.add(GlobalAveragePooling1D())#10
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))#12
    
    model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model=cnnmodel()
model.summary()

## start training
history = model.fit(X_train_reshape,y_train,epochs=50,batch_size=128, validation_split=0.1)
history.history
print(history.history["accuracy"])
print(history.history["val_accuracy"])

score = model.evaluate(X_test_reshape,y_test)

#%%####################################### 2D NN 
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow
from keras import layers
from scipy import signal

#%%###################################### preprocess the data
tmin = 0
tmax = 0.13
fmin = 50
fmax = 150
sfreq = 5000

# wavelet transform or stft
# f, t, Zxx = signal.stft(
#     ba_audio,
#     fs = 44100,
#     nfft=None,
#     noverlap=32,
#     nperseg=200,
#     window="boxcar",
#     )

# plt.figure()
# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.ylim([0,2500])
# plt.show()

X_tf = []

for c,x in enumerate(X):#loop trials
    Wx, scales = cwt(x, 'morlet')
    Wx=np.abs(Wx)
    Wx=(Wx - np.min(Wx)) / (np.max(Wx) - np.min(Wx)) # rescale
    X_tf.append(Wx)
X_tf = np.asarray(X_tf)

## train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tf, y, test_size = 0.20, random_state = 42)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# convert class vectors to binary class matrices
num_classes = 3
input_shape = X_train[0,:,:,:].shape
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        # CNN Block 1
        layers.Conv2D(filters=32,
                      kernel_size=(3, 3),
                      activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # CNN Block 2
        layers.Conv2D(filters=64,
                      kernel_size=(3, 3),
                      activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # CNN Block 3
        layers.Conv2D(filters=128,
                      kernel_size=(3, 3),
                      activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Dense Block
        layers.Flatten(),
        layers.Dense(num_classes,
                     activation="softmax"),
    ]
)

model.summary()

#%%####################################### Pretrained networks 
# Load the pre-trained EfficientNet B0 model
from tensorflow.keras.applications import EfficientNetB0

fakergb = tf.convert_to_tensor(X_train, np.float32)
fakergb = tf.image.grayscale_to_rgb(fakergb)
fakergb.shape
input_shape = fakergb[0,:,:,:].shape
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)

for layer in base_model.layers:
  layer.trainable = False
x = layers.Flatten()(base_model.output)
x = layers.Dense(1000, activation='relu')(x)
predictions = layers.Dense(num_classes, activation = 'softmax')(x)

## Training and testing 
batch_size = 1
epochs = 5
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

epochs = range(1, len(history.history["accuracy"]) + 1)

plt.plot(epochs, history.history["accuracy"], 'y', label='Training Accuracy')
plt.plot(epochs, history.history["val_accuracy"], 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize='pred')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=np.arange(3)+1,
                               )
disp.plot()

#%%####################################### 2D NN


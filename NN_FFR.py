#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:06:27 2024

@author: tzcheng
"""
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras import backend as K


## Load the data
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

## preprocess the data

## build the model
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

        # Dense Block
        layers.Flatten(),
        layers.Dense(num_classes,
                     activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
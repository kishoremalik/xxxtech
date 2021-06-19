#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pathlib import Path
import os.path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,Sequential
import tensorflow as tf


#train_dir = Path('/train')
#train_filepaths = list(dir_.glob(r'**/*.jpg'))

#test_dir = Path('/test')
#test_filepaths = list(dir_.glob(r'**/*.jpg'))

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=20,
                                   horizontal_flip = True)


training_set=train_datagen.flow_from_directory(
    directory="train",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=10,
    class_mode="binary",
    shuffle=True,
    seed=0)

print("train shape==",len(training_set))
test_datagen = ImageDataGenerator(rescale = 1./255)
testData=test_datagen.flow_from_directory(directory="valid", target_size=(224, 224),color_mode="rgb",batch_size=10,class_mode="binary",shuffle=True,seed=0)
input_shape=training_set.image_shape
print("train data###",input_shape)
print("test data###",len(testData))
IMG_SHAPE = (224,224,3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False, 
                                               weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1, activation='sigmoid')])

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
model.fit(training_set,epochs =20,validation_data=testData,verbose=2)
print("model training completed")
loss,accuracy=model.evaluate(testData)

print("accuracy=",accuracy)
print("loss=",loss)

model.save('document.h5')
print("model saved")


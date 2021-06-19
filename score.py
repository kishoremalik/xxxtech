#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras.models import load_model
from azureml.core.model import Model
from azureml.core import workspace
from azureml.core import Run
import os
import json
import joblib
from PIL import Image
import tensorflow as tf 

def init():
    global model
    model_path=os.path.join(os.getenv('AZUREML_MODEL_DIR'),'document.h5')
    model = load_model(model_path)
    print("model=",model)
def run(raw_data):
    value=None
    new_image = Image.fromarray(np.array(json.loads(raw_data), dtype='uint8'))
    imgv=tf.keras.preprocessing.image.img_to_array(new_image, data_format=None, dtype=None)
    imn=tf.image.resize(imgv,(224, 224))
    img = np.expand_dims(imn, axis=0)
    prediction = model.predict(img, batch_size=None,steps=1)
    print("prediction Value=",prediction[:,:])
    if(prediction[:,:]>0.5):
        value ="straightImage"
    else:
        value ="tiltedImage"
    return value


# In[ ]:





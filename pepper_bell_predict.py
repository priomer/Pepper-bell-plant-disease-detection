# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 23:22:56 2021

@author: ocn
"""

#prediction by pepper_bell1.py

from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    
    return img_tensor
    
model = load_model("model_pepper_bell_plant_disease.h5")
img_path = 'E:/AVRN_Report/PlantVillage/pepper_bell_whole/test/29.JPG'
check_image = load_image(img_path)
prediction = model.predict(check_image)
print(prediction)

prediction =np.argmax(prediction, axis=1)
if prediction==0:
    prediction="Bacterial_spot"
else:
    prediction="Healthy"

print(prediction)    
    
    

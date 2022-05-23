#!/usr/bin/env python
# coding: utf-8

# # Read the dataset 

# In[2]:


import numpy as np
import os
import cv2
import PIL
import PIL.Image
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from skimage.feature import hog

cwd = os.getcwd()
dataset_root_url = cwd + "\Data"
IMG_WIDTH=64  #**************************************hyperparameter
IMG_HEIGHT=64
img_data_array=[]
class_name=[]

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)      #**************************************hyperparameter
cells_per_block = (2, 2)



def read_dataset(dataset_root_url):
    for dir_level_1 in os.listdir(dataset_root_url):    #['vehicles', 'non-vehicles']
        for dir_level_2 in os.listdir(os.path.join(dataset_root_url, dir_level_1)):   #['.DS_Store', 'GTI_Far', 'GTI_Left', 'GTI_MiddleClose', 'GTI_Right', 'KITTI_extracted']
            if (dir_level_2 != '.DS_Store'):
                for file in os.listdir(os.path.join(dataset_root_url, dir_level_1,dir_level_2)):
                    if(file.lower().endswith(('.png', '.jpg', '.jpeg'))):
                        image_path= os.path.join(dataset_root_url, dir_level_1,dir_level_2, file)
                        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                        image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                        image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                        
                        image = hog(image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
                        
       
                        image=np.array(image) 
                        image = image.astype('float32')   
                        img_data_array.append(image)
                        class_name.append(dir_level_1)
    return img_data_array, class_name
 
image_data, class_name = read_dataset(dataset_root_url)  
#print(class_name[0])


# # Create the model

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.svm  import LinearSVC 
from sklearn.preprocessing import StandardScaler
import sklearn.externals as extjoblib
import joblib

image_data = StandardScaler(with_mean=0, with_std=1).fit_transform(image_data)
img_data_train, img_data_test, class_name_train, class_name_test = train_test_split(image_data, class_name , test_size=0.2, random_state=0)

#model = LogisticRegression() #********************************************
model = LinearSVC()

model.fit(img_data_train, class_name_train)

predictions = model.predict(img_data_test)
from sklearn.metrics import confusion_matrix



TN, FP, FN, TP = confusion_matrix(class_name_test, predictions).ravel()

print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)

accuracy =  (TP+TN) /(TP+FP+TN+FN)

print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))


# Save the Model
joblib.dump(model, 'LinearSVC.npy')


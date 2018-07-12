
# coding: utf-8

# ## Importing libraries and defining relevant folders

# #### Importing libraries

# In[1]:


import os, sys, time, datetime, glob, bcolz, random


# In[2]:


import shutil


# In[3]:


import keras
import numpy as np
import pandas as pd


# In[4]:


from keras import Sequential


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Convolution2D, Input, BatchNormalization


# In[7]:


from keras_vggface.vggface import VGGFace


# #### Relevant folders

# In[15]:


# In[16]:


folder_affectnet = "/pylon5/ir5fpcp/amogh112/AffectNet/"


# In[22]:


folder_train_data = folder_affectnet + 'data_copy/Manually_Annotated_Images/'


# ## Generating labels for valence and arousal training

# #### Reading and filtering dataframes from csv files

# In[18]:


path_training_csv = folder_affectnet + "Manually_Annotated_file_lists/training_va.csv"
folder_annotated_training_images = folder_affectnet + "Manually_Annotated_Images/"


# Getting image_filename and the corresponding labels for valence and arousal

# In[19]:


df_train = pd.read_csv(path_training_csv,header=None)
# df_train.head()


# In[23]:


list_image_paths = df_train.iloc[:,0].values


# In[24]:


df_va_train = df_train.iloc[:,[0,len(df_train.columns)-2,len(df_train.columns)-1]]
df_va_train.columns = ["image","valence", "arousal"]
print("shape is: ", df_va_train.shape)
# (df_va_train.head())


# Converting the dataframe in a dictionary

# In[25]:


dict_im_va = df_va_train.set_index('image').T.to_dict('list')
# dict_im_va


# #### Moving the files with bad labels from the training set

# In[26]:




# #### Defining function to get labels

# In[28]:


def get_va_from_image(image_name, dict_im_va=dict_im_va):
    va = dict_im_va.get(image_name)
    if va==None:
        return(np.zeros(2))
    else:
        return va


# ## Reading images from folder and saving convolutional features (for now just training)

# In[29]:


from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


# In[30]:


def get_data(path, target_size=(224,224), class_mode=None, shuffle=False, batch_size=10):
    gen = ImageDataGenerator()
    batches = gen.flow_from_directory(path, target_size=target_size, class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
    return batches


# In[31]:


# ## Defining custom flow_from_directory generator for labels
batches_flow_from_dir = get_data(folder_train_data)
# In[32]:


def flow_from_directory_va(flow_from_directory_gen):
    for x_batch in flow_from_directory_gen:
        idx = (flow_from_directory_gen.batch_index - 1) * flow_from_directory_gen.batch_size
        filenames_batch = flow_from_directory_gen.filenames[idx : idx + flow_from_directory_gen.batch_size]
        labels_batch = np.array([get_va_from_image(f) for f in filenames_batch])
        x_batch = preprocess_input(x_batch)
        yield x_batch, labels_batch


# In[33]:


list_train_image_fnames = (glob.glob(folder_train_data+'/*/*'))
# list_train_image_fnames

# # Models - Defining, loading data, training, saving

# ### Model 1: VGG16 -> model_vgg_reg1

# #### Defining the model

# In[34]:


vgg_full = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))


# In[35]:


# Building the model on top of the conv part

# In[36]:


p = 0.6


# In[40]:


inp_top_model = Input(vgg_full.layers[-1].output_shape[1:])
# x = BatchNormalization(axis=1)(inp_top_model)
x = Dropout(p/4)(inp_top_model)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_va = Dense(2, name='bva')(x)


# In[41]:


top_model = Model(inp_top_model, x_va)


# In[42]:


final_output = top_model(vgg_full.output)


# In[43]:


model_vgg_reg1 = Model(vgg_full.input, final_output)


# In[44]:


model_vgg_reg1.compile(loss='mean_squared_error',optimizer=optimizers.SGD(lr=0.001), metrics=['accuracy'])


# In[45]:

gen_va = flow_from_directory_va(batches_flow_from_dir)


# #### Training of the model

# In[47]:


model_vgg_reg1.fit_generator(gen_va, epochs=3, steps_per_epoch=10000)

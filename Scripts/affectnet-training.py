
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


get_ipython().system('ls $SCRATCH/AffectNet/data_copy')


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
df_train.head()


# In[23]:


list_image_paths = df_train.iloc[:,0].values


# In[24]:


df_va_train = df_train.iloc[:,[0,len(df_train.columns)-2,len(df_train.columns)-1]]
df_va_train.columns = ["image","valence", "arousal"]
print("shape is: ", df_va_train.shape)
(df_va_train.head())


# Converting the dataframe in a dictionary

# In[25]:


dict_im_va = df_va_train.set_index('image').T.to_dict('list')
dict_im_va


# #### Moving the files with bad labels from the training set

# In[26]:


image_name_bad_labels = ([t for t in dict_im_va.keys() if dict_im_va.get(t) == [-2.,-2.]])
len(image_name_bad_labels)


# Code to remove images with bad labels: 
folder_bad_labelled_train = folder_affectnet + "bad_label/"
if not os.path.exists(folder_bad_labelled_train):
    os.makedirs(folder_bad_labelled_train)
for im in image_name_bad_labels:
    src_path = folder_train_data+im
    dest_path = folder_bad_labelled_train+im
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
# In[27]:


len(dict_im_va)


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


batches_flow_from_dir = get_data(folder_train_data)


# ## Defining custom flow_from_directory generator for labels

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
list_train_image_fnames

#Tracking the unlabelled images
a=list_train_image_fnames[0]
a

os.path.basename(os.path.split(a)[0])+'/'+(os.path.split(a)[1])

l = ([(os.path.basename(os.path.split(f)[0])+'/'+(os.path.split(f)[1])) for f in list_train_image_fnames])

a =list(map(get_va_from_image,l))

len(a)

a.count(None)
# # Models - Defining, loading data, training, saving

# ### Model 1: VGG16 -> model_vgg_reg1

# #### Defining the model

# In[34]:


vgg_full = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))


# In[35]:


vgg_full.summary()


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


model_vgg_reg1.summary()


# #### Loading batches, making destination folders
batches_flow_from_dir = get_data(folder_train_data)list_filenames_affectnet = batches_flow_from_dir.filenames
print("total number of files are: ", len(list_filenames_affectnet))
# In[46]:


gen_va = flow_from_directory_va(batches_flow_from_dir)


# #### Training of the model

# In[47]:


model_vgg_reg1.fit_generator(gen_va, epochs=3, steps_per_epoch=10000)


# In[137]:


new_model.summary(ImageDataGenerator())


# #### Unnecessary for now: saving conv features for VGG16

# Saving conv features
folder_affectnet_conv_features_vgg16 = folder_affectnet + "features/vgg16/conv_features"
if not os.path.exists(folder_affectnet_conv_features_vgg16):
    os.makedirs(folder_affectnet_conv_features_vgg16)
# In[40]:


layers = model_vgg16.layers
layer_idx = [index for index,layer in enumerate(layers) if type(layer) is Convolution2D][-1]
conv_layers, fc_layers = layers[:layer_idx+1], layers[layer_idx+1:]


# In[41]:


model_vgg16_conv = Sequential(conv_layers)


# In[42]:


model_vgg16_conv.layers


# ## Model 2: 

# In[72]:


vgg_face_full = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg')


# In[73]:


vgg_face_full.summary()


# ### Model 2 - MobileNet

# In[15]:


mobnet = applications.MobileNet()


# In[19]:


mobnet.summary()


print("loading libraries")
import os, sys, random, glob, argparse, math, gc

import cv2
import dlib
import imutils
from imutils import face_utils
import matplotlib
# import matplotlib.pyplot as plt #causes segmentation fault, so do not use.
from skimage.feature import hog
from skimage import exposure

import sklearn
from sklearn import svm, metrics
import pickle

import numpy as np
import pandas as pd
from bcolz import carray

from tqdm import tqdm
from time import sleep
import datetime as dt
print("libraries loaded")

folder_DISFA_data = "/pylon5/ir5fpcp/amogh112/DISFA_data/"
folder_DISFA_FAU = "/pylon5/ir5fpcp/amogh112/DISFA_data/ActionUnit_Labels/"
folder_DISFA_FAU_summary = "DISFA_FAUs/"

# returns a dictionary in the form: {'SN001':{'positives': [1,2,3],'negatives':[4,5,6,7] }}
# ie corresponding to each subject a dictionary which contains list frame nos which are positives and 
def getDISFAFramesDictionary(folder_DISFA_FAU_summary, fau_no, fau_thresh):
    df_fau = pd.read_csv(folder_DISFA_FAU_summary + "{}/".format(fau_thresh) + "FAU{}.csv".format(fau_no))
    df_positives = df_fau.filter(regex="^((?!neg).)*$",axis=1)
    df_negatives = df_fau.filter(like="neg",axis=1) 
    list_subjects = df_positives.columns.values
    fau_dict = {}
    for subj in list_subjects:
        fau_dict[subj] = {'positives':[], 'negatives':[]}
        fau_dict[subj]['positives'] = [f for f in df_positives[subj].values if not math.isnan(f)]
        fau_dict[subj]['negatives'] = [f for f in df_negatives["{}_neg".format(subj)].values if not math.isnan(f)]
    return fau_dict

def equaliseDictionary(fau_dict):
    for subj in fau_dict.keys():
        number_positives = len(fau_dict[subj]['positives'])
        number_negatives = len(fau_dict[subj]['negatives'])
        if number_negatives >= number_positives:
            fau_dict[subj]['negatives'] = random.sample(fau_dict[subj]['negatives'], number_positives)
        else:
            fau_dict[subj]['positives'] = random.sample(fau_dict[subj]['positives'], number_negatives)
    return fau_dict

# returns a dictionary with keys as fold_0,fold_1,...,test
# make sure number of folds exactly divide the train subjects
def getTrainTestFolds (fau_dict, no_folds, no_test_subjects):
    list_subjects = fau_dict.keys()
    no_train_subjects = len(list_subjects) - no_test_subjects
    random.shuffle(list_subjects)
    test_subjects = list_subjects[-no_test_subjects:]
    train_subjects = list_subjects[:-no_test_subjects]
    dict_folds = {'test':{}}
    # putting train and test subjects in new dictionary
    for subj in test_subjects:
        dict_folds['test'][subj] = fau_dict[subj]
    fold_size = no_train_subjects / no_folds
#     fold_size_remainder = no_train_subjects % no_folds
    for fold_no in range(no_folds):
        fold_subjects = train_subjects[fold_no*fold_size : fold_no*fold_size+fold_size]
        dict_folds ['fold_{}'.format(fold_no)]={}
        for sub in fold_subjects:
            dict_folds ['fold_{}'.format(fold_no)] [sub] = fau_dict [sub]
    return dict_folds


# ### Crop and save images and features

# ##### Function for cropping given an image path 

def similarityTransform(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform;

#new function, doesnt write landmarks every single time
def detectAndaligncrop(impath, detector, predictor):
    image=cv2.imread(impath)
    image_float=np.float32(image)/255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    #initialising images and allPoints arrays
    allPoints=[]
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        points=[]
        for (x,y) in shape:
            points.append((x,y))
        allPoints.append(points)
    images=[image_float]
    #computation
    w=112
    h=112
    eyecornerDst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ];
    imagesNorm = [];
    pointsNorm = [];
    #     print allPoints[0]
    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ]);
    n = len(allPoints[0]);
    numImages = len(images)
    for i in xrange(0, numImages):
        points1 = allPoints[i];
        # Corners of the eye in input image
        eyecornerSrc  = [ allPoints[i][36], allPoints[i][45] ] ;
        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst);
        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w,h));
    #         print("debug im type shape max mean min ", img.dtype,img.shape,np.max(img),np.mean(img),np.min(img))
    #         plt.imshow(img)
        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68,1,2));        
        points = cv2.transform(points2, tform);
        points = np.float32(np.reshape(points, (68, 2)));
        pointsNorm.append(points);
        imagesNorm.append(img);
    #     print (pointsNorm[0])
    #     plt.imshow(imagesNorm[0]) 
    # Output image
    output=imagesNorm[0]
    rgb_image=cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    return rgb_image, pointsNorm[0]


# ##### Functions for getting features

# Getting HOG, given an image path or an image, return features

#takes in rgb images and returns the required HOG descriptor array. 
def getHOGFeatures (orientations, pixels_per_cell, cells_per_block, image):
    if isinstance(image, basestring):
        im = cv2.cvtColor(cv2.imread(image),cv2.COLOR_BGR2RGB)
    else:
        im = image
    gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) 
    fd, hog_image = hog(gray_im, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=True)
#     hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#     plt.imshow (hog_image_rescaled, cmap = plt.cm.gray)
#     print("HOG vector dimension: ", fd.shape)
    return fd

# ##### Preprocessing functions and function_dictionary

def FAU4_1(image,landmarks):
    cropped_im=image[:38]
    return cropped_im

def FAU1_1(image,landmarks):
    cropped_im=image[:38]
    return cropped_im

def FAU2_1(image,landmarks):
    cropped_im=image[:38]
    return cropped_im

def FAU5_1(image,landmarkPoints): #includes border
    rect_top=int(landmarkPoints[17][1])
    rect_bottom=int(landmarkPoints[29][1])
    rect_left=int(landmarkPoints[3][0])
    rect_right=int(landmarkPoints[12][0])
    cropped_im=image[rect_top:rect_bottom,rect_left:rect_right]
    border_top, border_bottom, border_left, border_right = [0,32-height,0,64-width]
    img_with_border = cv2.copyMakeBorder(cropped_im, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_with_border

def FAU12right_1(image,landmarkPoints):
    rect_top = int(landmarkPoints[34][1])
    rect_bottom = int(landmarkPoints[11][1])
    rect_left = int(landmarkPoints[34][0])
    rect_right = int(landmarkPoints[11][0])
    cropped_im = image[rect_top:rect_bottom,rect_left:rect_right]
    border_top, border_bottom, border_left, border_right = [0,32-height,0,32-width]
    img_with_border = cv2.copyMakeBorder(cropped_im, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_with_border

def FAU12left_1(image,landmarkPoints):
    rect_top = int(landmarkPoints[32][1])
    rect_bottom = int(landmarkPoints[5][1])
    rect_left = int(landmarkPoints[5][0])
    rect_right = int(landmarkPoints[32][0])
    cropped_im = image[rect_top:rect_bottom,rect_left:rect_right]
    border_top, border_bottom, border_left, border_right = [0,32-height,0,32-width]
    img_with_border = cv2.copyMakeBorder(cropped_im, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_with_border

function_dict={'FAU1_1':FAU1_1,'FAU2_1':FAU2_1,'FAU4_1':FAU4_1,'FAU5_1':FAU5_1, 'FAU12right_1':FAU12right_1, 'FAU12left_1':FAU12left_1}


# ##### Crop and save function

# Made by keeping in mind that these are the parameters that we need to pass: o, ppc cpb, fau_no, thresh, function used for cropping, folders

#saves images and HOG features given the o,ppc,cpb,fau_no,thresh,dict_folds,cropping_function_name,function_dict in folder_DISFA_data/thresh/cropping_function_name
def cropAndSaveImageHOG (o ,ppc ,cpb ,fau_no , thresh, dict_folds, folder_DISFA_data, cropping_function_name, function_dict, featuresFunction, boolSave=True):
    folder_cropped_images = folder_DISFA_data + "/features/cropped_images/"
    folder_dest = folder_cropped_images +  "/{}/{}/".format(thresh,cropping_function_name)
    folder_features_dest = folder_DISFA_data + "/features/hog/{}/{}/".format(thresh,cropping_function_name)
    print("images go to: ",folder_dest, "\n", "features go to:", folder_features_dest)
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(folder_dest):
        os.makedirs(folder_dest)
    if not os.path.exists(folder_features_dest):
        os.makedirs(folder_features_dest)
    for fold in dict_folds.keys():
        print ("inside:", fold)
        for subj in dict_folds[fold]:
            print ("saving subject: ", subj)
            for category in dict_folds[fold][subj]:
                print("Images in this category are : ", len(dict_folds[fold][subj][category]))
#                 print ("inside: ",fold,subj,category)
                folder_dest_image = folder_dest + "{}/{}/{}/".format(fold,subj,category)
                folder_dest_feature = folder_features_dest + "{}/{}/{}/".format(fold,subj,category)
                if not os.path.exists(folder_dest_image):
                    os.makedirs(folder_dest_image)
                for frame_no, frame in enumerate(dict_folds[fold][subj][category]):
                    im_path = folder_DISFA_data + "Videos_RightCamera/RightVideo{}/{}.jpeg".format(subj,int(frame))
                    im_basename = os.path.basename(im_path)
                    im_dest_path = folder_dest_image + im_basename 
                    features_path = folder_dest_feature + os.path.splitext(im_basename)[0] 
                    if os.path.exists(im_path):
                        try:
                            #cropping and aligning images
                            im_aligned_cropped,landmarkPoints = detectAndaligncrop(im_path, detector, predictor)
                            cropped_rgb_image = function_dict[cropping_function_name] (im_aligned_cropped, landmarkPoints)
                            #saving cropped RGB images in BGR(because opencv uses BGR as default)
                            cv2.imwrite(im_dest_path, cv2.cvtColor(cropped_rgb_image,cv2.COLOR_RGB2BGR)*255.)
                            #getting features
                            fd = featuresFunction(o, ppc, cpb, cropped_rgb_image)
                            #saving features
                            if not (os.path.exists(features_path)):
                                os.makedirs(features_path)
                            carray_fd = carray(fd, rootdir=features_path, mode = 'w')
                            carray_fd.flush()
                        except KeyboardInterrupt:
                            break
                        except: 
                            continue
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
       


# ##### Final abstract function to crop and save images
# Inputs are: o, ppc, cpb, fau_no, thresh, cropping_function_name <br>
# Optional inputs: no_folds=5, no_test_subjects=2, function_dict=function_dict, featuresFunction=getHOGFeatures, folder_DISFA_FAU_summary=folder_DISFA_FAU_summary, folder_DISFA_data=folder_DISFA_data, boolEqualise=True


def finalSaveImagesFeatures(o ,ppc ,cpb ,fau_no , thresh, cropping_function_name, no_folds=5, no_test_subjects=2, function_dict=function_dict, featuresFunction=getHOGFeatures, folder_DISFA_FAU_summary=folder_DISFA_FAU_summary, folder_DISFA_data=folder_DISFA_data, boolEqualise=True):
    frames_dict = getDISFAFramesDictionary(folder_DISFA_FAU_summary,fau_no,thresh)
    frames_dict = equaliseDictionary(frames_dict)
    dict_folds = getTrainTestFolds(frames_dict,no_folds,no_test_subjects)
    cropAndSaveImageHOG(o ,ppc ,cpb ,fau_no ,thresh , dict_folds, folder_DISFA_data,cropping_function_name,function_dict,getHOGFeatures)


# ### trainSVMGridSearchModel helper function
# Using GridSearchCV model

# ### Train function to use custom cross validation generator

# Helper function to train once custom iterable, train and the test function have been defined.

def trainSVMGridSearchModel(X_train, Y_train , custom_fold_iterable, no_jobs=1, kernel_list=['rbf','linear']):
    #setup parameter search space
    gamma_range = np.outer(np.logspace(-3,0,4),np.array([1,5]))
    gamma_range = gamma_range.flatten()
    C_range = np.outer(np.logspace(-1,1,3),np.array([1,5]))
    C_range = C_range.flatten()
    parameters = {'kernel': kernel_list,'C':C_range,'gamma':gamma_range}
    svm_clsf = svm.SVC()
    grid_clsf = sklearn.model_selection.GridSearchCV(estimator=svm_clsf,param_grid=parameters,n_jobs=no_jobs,verbose=2,cv=custom_fold_iterable)
    #train
    start_time=dt.datetime.now()
    print('Start param searching at {}'.format(str(start_time)))
    grid_clsf.fit(X_train,Y_train)
    elapsed_time=dt.datetime.now()-start_time
    print('Elapsed time, param searching {}'.format(str(elapsed_time)))
    sorted(grid_clsf.cv_results_.keys())
    return grid_clsf


def trainCustomGridSearch(fau_no, thresh, cropping_function_name ,trainFunction , no_jobs=8, folder_data=folder_DISFA_data):
    
    fold_folder_list = glob.glob(folder_data + "features/hog/{}/{}/*".format(thresh,cropping_function_name))
    
    # defining global holders and variables
    no_folds = len(fold_folder_list)
    features = []
    targets = []
    fold_label_list = []

    #processing for each fold:
    for fold_no, fol in enumerate(fold_folder_list):
        
        #lists specific to fold
        list_positive_feature_folders = []
        list_negative_feature_folders = []
        positive_features = []
        negative_features = []
        fold_targets = []
        fold_train_features = []
        
        #loading features in lists
        list_positive_feature_folders.extend(glob.glob(fol + "/*/positives/*/"))
        list_negative_feature_folders.extend(glob.glob(fol + "/*/negatives/*/"))
        print("loading positive features for fold: ", fold_no)
        for pos_feat_folder in list_positive_feature_folders:
            pos_feat = carray(rootdir = pos_feat_folder, mode = 'r')
            positive_features.append(pos_feat)
        print("loading negative features for fold: ", fold_no)
        for neg_feat_folder in list_negative_feature_folders:
            neg_feat = carray(rootdir = neg_feat_folder, mode = 'r')
            negative_features.append(neg_feat)

        fold_train_features.extend(positive_features)
        fold_train_features.extend(negative_features)
        fold_targets.extend([1] * len(positive_features))
        fold_targets.extend([0] * len(negative_features))
        no_fold_features = len(positive_features) + len(negative_features)
        print("this fold has these many features: ",no_fold_features)
        
        #updating global features and targets
        features.extend(fold_train_features)
        targets.extend(fold_targets)
        #updating fold_label_list
        fold_label_list.extend([fold_no]*no_fold_features)

    #defining the custom cross validation generator over training data
    cvIterable= []
    for fold_no in range(no_folds):
        fold_label_list = np.array(fold_label_list)
        train_indices = np.argwhere(fold_label_list != fold_no).flatten()
        test_indices = np.argwhere(fold_label_list == fold_no).flatten()
        cvIterable.append((train_indices,test_indices))
    
    classifier_results = trainSVMGridSearchModel(features ,targets, cvIterable ,no_jobs=no_jobs , kernel_list=['linear'])
    
    return classifier_results 


def saveClassifierResults(fau_no, thresh, cropping_function_name ,trainFunction , no_jobs=8, folder_data=folder_DISFA_data):
    
    print("training result saved") 
    grid_search_result = trainCustomGridSearch(fau_no, thresh, cropping_function_name ,trainFunction , no_jobs=no_jobs, folder_data=folder_data)
    print("training done")
    
    best_classifier = grid_search_result.best_estimator_
    params = grid_search_result.best_params_
    
    folder_model_dest = "{}/models/{}_{}/".format(folder_data,thresh, cropping_function_name)
    if not os.path.exists(folder_model_dest):
        os.makedirs(folder_model_dest)
    file_result_dump  = folder_model_dest + "result.sav"
    file_model_dump  = folder_model_dest + "best_model.sav"
    
    pickle.dump(grid_search_result,open(file_result_dump,'wb'))
    pickle.dump(best_classifier,open(file_model_dump,'wb'))
    print("best params are: ", params)



 ### Example functions to save and crop images; execute to process train

# In[2
if __name__ == '__main__':
#        print("saving FAU2 thresh 2")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,2 , 2, 'FAU2_1')
#        print("saving FAU2 thresh 3")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,2 , 3, 'FAU2_1')
#        print("saving FAU4 thresh 2")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,4 , 2, 'FAU4_1')
#        print("saving FAU4 thresh 3")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,4 , 3, 'FAU4_1')
 #       print("saving FAU1 thresh 2")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,1 , 2, 'FAU1_1')
#        print("saving FAU1 thresh 3")
#	finalSaveImagesFeatures (6 ,(8,8) ,(4,4) ,1 , 3, 'FAU1_1')
#        saveClassifierResults(2,2,'FAU2_1',1,28)
        saveClassifierResults(2,3,'FAU2_1',1,28)
        saveClassifierResults(4,2,'FAU4_1',1,28)
#        saveClassifierResults(4,3,'FAU4_1',1,28)
#        saveClassifierResults(1,2,'FAU1_1',1,28)
#        saveClassifierResults(1,3,'FAU1_1',1,28)
        print("done")

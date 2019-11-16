'''
Shared general functions and warnings
'''

import numpy as np
import torch

import warnings
#I made modifications to the code of the model after training
#These don't alter the output, but were just clean up
#The user warning is just notifying changes to the interpolation function
warnings.filterwarnings("ignore", category=torch.serialization.SourceChangeWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

import os

from embiggen.io import all_scenes_paths
from embiggen.aggregate import baseline_upscale

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def runData(model, data):
    '''
    Processes the image with the provided model
    ______
    In:
    model:  The model to be used
    data:  A single image of (1,128,128) shape
    Out:
    result:  The SR image
    '''

    d = np.expand_dims(data, axis=0)
    image_tensor = torch.from_numpy(d.copy()).float().to(device)
    with torch.no_grad():
        output = model(image_tensor)
        result = output.cpu().data.numpy()
        result = np.resize(result, (384,384))
    return result

def ensemble(images1, images2):
    '''
    Merges two models together via np.mean()
    _______
    In:
    images1: The first set of images from model1
    images2: The second set of images from model2
    Out:
    images:  An array of the np.mean() images across the two models
    '''

    images = []
    for i in range(len(images1)):
        holder = np.zeros((2,384,384))
        holder[0,:,:] = images1[i]
        holder[1,:,:] = images2[i]
        resultant = np.mean(holder, axis=0)
        images.append(resultant)
    return images

def bicubicUpsample(path_to_data, n=-1):
    '''
    Wrapper to perform bicubic interpolation per the baseline on all images
    ______
    Out:
    images:  Array of images resized to (384,384) with bicubic interpolation
    '''

    images = []
    total = len(path_to_data)
    if n > 0:
        total = n
    i = 0
    for d in all_scenes_paths(path_to_data):
        images.append(baseline_upscale(d))
        i += 1
        if i > total:
            break
    return images

def makeDirectories():
    '''
    Makes sure all the directories exist, if not makes them exist
    '''

    directories = ['models', 'logs', 'previews', 'model_state_dict']

    for d in directories:
        if not os.path.isdir(d):
            os.mkdir(d)
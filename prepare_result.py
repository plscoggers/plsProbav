'''
Functions used to prepare the result for the probav super resolution challenge
Only used for test set data
'''

import os

import torch
import numpy as np 

from embiggen.io import all_scenes_paths, prepare_submission
import probav_data as pd
import resnet_test
import utils
from utils import bicubicUpsample, ensemble, device, runData

utils.makeDirectories()
path_to_test = 'probav_data/test'

def predictTest(model_file):
    model = torch.load(model_file).to(device)
    data_paths = all_scenes_paths(path_to_test)
    dataset = pd.ProbaVDataset(data_paths, True)
    images = []
    model.eval()
    for d in dataset:
        images.append(runData(model, d))
    return images

def run(model_file, ensemble_bicubic=True):
    '''
    Main entry point
    Runs all the images in the test set, ensembles if requested with the bicubic baseline.
    _____
    In:
    model_file: path to the model
    ensemble_bicubic:  whether or not to ensemble the results with bicubic interpolation

    Out:
    results:  a list of the resultant images
    '''

    print('Running model on test set')
    model_images = predictTest(model_file)
    if ensemble_bicubic:
        print('Getting bicubic samples')
        bicubic_images = bicubicUpsample(path_to_test)
        print('Ensembling the two models')
        results = ensemble(model_images, bicubic_images)
    else:
        results = model_images
    prepare_submission(results, all_scenes_paths(path_to_test))
    return results

if __name__ == '__main__':
    run('models/inference_306.pt', True)




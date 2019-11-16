'''
Contains general functions to run the model on training set data and obtain the score information.
Main runs trough all the models contained in the models directory and prints the returned score
dictionary.
'''

import os

import torch
import numpy as np

import resnet_test
import probav_data as pd
from embiggen.io import all_scenes_paths
from embiggen.score import score_image
import utils
from utils import device, runData

utils.makeDirectories()


def runToScore(model, dataset, data_paths, n=-1):
    '''
    Runs on n many train data and returns the score and the resultant image
    _______
    In:
    model: The model being used
    dataset: The ProbaVDataset
    data_paths:  List of paths of the train dataset
    
    Out:
    scores:  The scores vs the baseline
    results:  The resultant images
    '''

    scores = []
    results = []
    model.to(device)
    model.eval()
    if n < 0:
        n = len(dataset)
    counter = 0
    for d, _, _ in dataset:
        if counter < n:
            result = runData(model, d)
            score = score_image(result, data_paths[counter])
            scores.append(score)
            results.append(result)
            counter += 1
        else:
            break
    return scores, results

def multipleFiles(model_directory, data_type=pd.ProbaVDataset):
    '''
    Iterates through all the models in the directory and runs each one
    _____
    In:
    model_directory:  Where the models are stored
    data_type:  Leave as default, previously used to test other dataset models
    
    Out:
    scores:  The average score for each model in dictionary form
    '''
    scores = []
    for m in os.listdir(model_directory):
        model = torch.load(model_directory + m).to(device)
        data_paths = all_scenes_paths('probav_data/train')
        dataset = data_type(data_paths)
        score = run(model, dataset, data_paths)
        scores.append({'score': score, 'model': m})
    return scores

if __name__ == '__main__':
    print(multipleFiles('models/'))



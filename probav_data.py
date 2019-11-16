import torch
from torch.utils import data
import numpy as np
from embiggen.io import highres_image, lowres_image_iterator


class ProbaVDataset(data.Dataset):
    '''
    A torch Dataset wrapper for our data
    ____
    In:
    data_dict: This is a dictionary output from the function in train.py getFullTrainingInfo
    test: Whether or not we are training or testing
    '''

    def __init__(self, data_dict, test=False):
        self.data_dict = data_dict
        self._test = test
    
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        if self._test:
            return self._getImagesTest(index)
        else:
            return self._getImages(index)

    def _augmentImage(self, img, aug_types):
        '''
        Applies augmentation
        _____
        In:
        img: The image to be augmented
        aug_types: Dictionary containing the type of augmentations to perform
        Out:
        temp_img: The augmented image
        '''

        temp_img = img

        if aug_types['flip'] == True:
            temp_img = np.flip(temp_img)

        for _ in range(aug_types['rotate']):
            temp_img = np.rot90(temp_img)

        return temp_img.copy()

    def _getGroundTruthInfo(self, index):
        '''
        Gets and augments the ground truth.  Test dataset only.
        I also pre-emptively apply the mask to the ground truth.
        '''

        info_dict = self.data_dict[index]

        ground_truth, target_map = highres_image(info_dict['path'])

        ground_truth = self._augmentImage(ground_truth, info_dict)
        target_map = self._augmentImage(target_map, info_dict)

        target_map = np.expand_dims(target_map,axis=0)
        ground_truth = np.expand_dims(ground_truth,axis=0)

        ground_truth = ground_truth * target_map

        return ground_truth, target_map

    def _processImages(self, info, augment):
        '''
        Gets, augments, and calculates median for all low resolution images.  
        Returns the median image.
        '''
        image_counter = 0
        images = []
        path_to_images = info
        if augment:
            path_to_images = info['path']
        for lr, _ in lowres_image_iterator(path_to_images):
            if augment:
                lr = self._augmentImage(lr, info)
            lr = np.expand_dims(lr, axis=0)
            images.append(lr)
        
        numpy_images = np.zeros((len(images), 1, 128, 128))
        for i in range(len(images)):
            numpy_images[i,:,:,:] = images[i]
        return np.median(numpy_images, axis=0)

    def _getImagesTest(self, index):
        path_to_images = self.data_dict[index]
        return self._processImages(path_to_images, False)

    def _getImages(self, index):
        '''
        Gets the image, and ground truth if test, and finds the median.
        ______
        In:
        index: The index of the training set you're looking for
        Out:
        A dictionary containing the median image.  If not test includes ground truth and mask
        '''

        info_dict = self.data_dict[index]
        ground_truth, target_map = self._getGroundTruthInfo(index)

        return self._processImages(info_dict, True), ground_truth, target_map
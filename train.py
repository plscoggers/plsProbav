'''
Builds and trains the model
'''

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from embiggen.io import all_scenes_paths
import resnet_test as rt
from probav_data import ProbaVDataset

import utils
from utils import device

np.random.seed(16)

DATA_PATH = 'probav_data/'

BATCH_SIZE = 16

total_passes = 0
epoch = 0

utils.makeDirectories()
logs_file = open('logs/logs.txt', 'w')

def getFullTrainingInfo(all_train, use_rotate=False, use_flip=False): 
    '''
    Used create the list of images to use.
    _____
    In:
    all_train:  A list of all the training info from the embiggen call
    use_rotate: Default False, will rotate the image for image augmentation
    use_flip: Default False, will flip the image 90 degrees 3 different times

    Out:
    all_train_dicts: A list of dicts containing the path to the images and 
    whether or not to augment the image
    '''

    all_train_dicts = []
    for train in all_train:
        all_train_dicts.append({'path': train, 'flip': False, 'rotate': 0})
        if use_rotate:  
            all_train_dicts.append({'path': train, 'flip': False, 'rotate': 1})
            all_train_dicts.append({'path': train, 'flip': False, 'rotate': 2})
            all_train_dicts.append({'path': train, 'flip': False, 'rotate': 3})
        if use_flip:
            all_train_dicts.append({'path': train, 'flip': True, 'rotate': 0})
        if use_flip and use_rotate:
            all_train_dicts.append({'path': train, 'flip': True, 'rotate': 1})
            all_train_dicts.append({'path': train, 'flip': True, 'rotate': 2})
            all_train_dicts.append({'path': train, 'flip': True, 'rotate': 3})
    return all_train_dicts

def cMSE(output, gt_map, target):
    '''
    cMSE calculation without shift
    _____
    In:
    output: The output of the network
    gt_map: The mask of the target
    target: The ground truth image

    Out:
    cMSE score
    '''

    mapped_output = output * gt_map  #Applies occlusions map
    diff = target - mapped_output    #Finds the difference between super resolution and target
    b = torch.mean(diff)             #Bright
    diff -= b                        #Subtract bright pixels
    return torch.mean(diff ** 2)     #mean MSE brightness adjusted

def preview(data, target, gt_map):
    '''
    Used to create previews of the network outputs
    _______
    In:
    data: The output of the network
    target: The ground truth
    gt_map: The occlusion map
    '''

    single_target_cpu = target.cpu().data.numpy()[0,:,:,:]
    single_target_cpu = np.reshape(single_target_cpu, (384,384))
    plt.imshow(single_target_cpu)
    plt.savefig('previews/target{}.png'.format(total_passes + 1))
    result = data.cpu().data.numpy()[0,:,:,:]
    result = np.reshape(result, (384,384))
    plt.imshow(result)
    plt.savefig('previews/{}.png'.format(total_passes + 1))
    plt.close()

def train(model, train_data, opt, epoch, preview_outputs=False):
    global total_passes  #This is just used to determine whether or not a preview should be made when preview is active
    model.train()

    epoch_loss = 0
    batch_counter = 0

    for data, target, gt_map in train_data:

        data = data.float().to(device)
        target = target.float().to(device)
        gt_map = gt_map.float().to(device)

        opt.zero_grad()

        output = model(data)

        loss = cMSE(output, gt_map, target)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #just to avoid gradient issues when training
        loss.backward()
        opt.step()

        logs_file.write('Batch: {}/{}, Epoch: {}, Total: {}, Batch Loss: {}\n'.format(batch_counter + 1, 
                                                                                      len(train_data), 
                                                                                      epoch, 
                                                                                      total_passes + 1, 
                                                                                      loss.item())
                                                                                      )
        print('Batch: {}/{}, Epoch: {}, Total: {}, Batch Loss: {}'.format(batch_counter + 1, 
                                                                          len(train_data), 
                                                                          epoch, 
                                                                          total_passes + 1, 
                                                                          loss.item())
                                                                          )
        
        if preview_outputs and (total_passes + 1) % 100 == 0:
            preview(output, target, gt_map, total_passes)

        batch_counter += 1
        total_passes += 1

def test(model, test_data):
    model.eval()
    loss = 0

    print('---Testing---')

    with torch.no_grad():
        for data, target, gt_map in test_data:
            data = data.float().to(device)
            target = target.float().to(device)
            gt_map = gt_map.float().to(device)

            output = model(data)

            loss += cMSE(output, gt_map, target).item()

    logs_file.write('Test MSE: {}'.format( loss / len(test_data)))
    print('Test MSE: {}'.format(loss / len(test_data)))

def run(model, train_data, test_data, opt, n_epochs=3):
    print(device)
    for i in range(n_epochs - epoch):
        print('Epoch {} Starting'.format(i + 1))   

        train(model, train_data, opt, i + 1)

        test(model, test_data)

        checkpoint = {
            'encoder': model.encoder, 
            'decoder': model.decoder,
            'encoder_state_dict': encoder.state_dict(), 
            'decoder_state_dict': decoder.state_dict(), 
            'optimizer': opt, 
            'optimizer_state_dict': opt.state_dict(),
            'epoch': i + 1, 
        }

        torch.save(checkpoint, 'model_state_dict/sr-probav_state_dict.pt')
        torch.save(model, 'models/inference_{}.pt'.format(i + 1))

if __name__ == '__main__':
    all_train = all_scenes_paths(DATA_PATH + 'train')

    all_train_info = getFullTrainingInfo(all_train, True, True)

    train_dataset, val_dataset = model_selection.train_test_split(all_train_info, test_size=0.1, shuffle=True)

    train_dataset = ProbaVDataset(train_dataset)
    val_dataset = ProbaVDataset(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    encoder = rt.ResnetBlocks(8, 64, 7)
    decoder = rt.DecodeResNet(64, 64, 1, 7)
    model = rt.CheesyResnetWDecoder(encoder, decoder)

    model.to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

    run(model, train_dataloader, val_dataloader, opt, 2000)
    

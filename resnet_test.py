import torch.nn as nn


def make_res_block(num_planes, norm=nn.BatchNorm2d):
    block = nn.Sequential(nn.Conv2d(num_planes, num_planes, kernel_size=3, padding=1),
                                norm(num_planes),
                                nn.LeakyReLU(),
                                nn.Conv2d(num_planes, num_planes, kernel_size=3, padding=1),
                                norm(num_planes)
                                )
    return block

class ResnetBlocks(nn.Module):
    '''
    Creates a series of resnet blocks to be used for encoding
    ______
    In:
    num_blocks: The number of resnet blocks to create
    num_planes: The number of filters for each resnet block
    init_conv_kernel: The size of the kernel for the initial convolution
    '''

    def __init__(self, num_blocks=8, num_planes=64, init_conv_kernel=7):
        super(ResnetBlocks, self).__init__()
        self.conv1 = nn.Conv2d(1, num_planes, kernel_size=init_conv_kernel, padding=init_conv_kernel // 2)
        self.relu = nn.LeakyReLU()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(make_res_block(num_planes))
  
    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)

        for layer in self.layers:
            res = x
            x = layer(x)
            x += res
        return x

class DecodeResNet(nn.Module):
    '''
    A simple interpolating decoder
    ______
    In:
    init_planes: Number of filters expected from input
    size_up_planes:  Number of filters for subsequent layers
    result_planes: Final result channel size
    final_conv_kernel:  Size of last convolution's kernel
    '''

    def __init__(self, init_planes=64, size_up_planes=64, result_planes=1,final_conv_kernel=5):
        super(DecodeResNet, self).__init__()
        self.conv1 = nn.Conv2d(init_planes, size_up_planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(size_up_planes, size_up_planes, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(size_up_planes, result_planes, kernel_size=final_conv_kernel, padding=final_conv_kernel // 2)
        self.dropout = nn.Dropout2d(0.2)
        self.relu = nn.LeakyReLU()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = nn.functional.interpolate(x, mode='bicubic', scale_factor=3)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.activation(x)
        return x

class CheesyResnetWDecoder(nn.Module):
    '''
    A wrapper for the encoder and decoder
    This is a bit redundant but I wanted to have a nice wrapper for the inference model output
    _____
    In:
    encoder: The encoder to be used
    decoder: The decoder to be used
    '''

    def __init__(self, encoder, decoder):
        super(CheesyResnetWDecoder, self).__init__()     
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
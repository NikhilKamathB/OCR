import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple
from .utils import InitializeModule


class VGG16_BN(torch.nn.Module):

    '''
        Custom VGG-16 with batch normalization.
    '''

    def __init__(self, freeze: bool = False) -> None:
        '''
            Initial definitions for the VGG-16 model.
            Input params: `freeze` - a boolean value indicating whether to freeze
                                     weights of first block.
            Returns: `None`.

        '''
        super().__init__()
        initialize_module = InitializeModule()
        vgg16_bn_features = models.vgg16_bn(weights="VGG16_BN_Weights.DEFAULT").features
        self.block_1 = nn.Sequential()
        self.block_2 = nn.Sequential()
        self.block_3 = nn.Sequential()
        self.block_4 = nn.Sequential()
        for i in range(13):
            self.block_1.add_module(f"BLOCK_1_{str(i)}", vgg16_bn_features[i])
        for i in range(13, 20):
            self.block_2.add_module(f"BLOCK_2_{str(i)}", vgg16_bn_features[i])
        for i in range(20, 30):
            self.block_3.add_module(f"BLOCK_3_{str(i)}", vgg16_bn_features[i])
        for i in range(30, 40):
            self.block_4.add_module(f"BLOCK_4_{str(i)}", vgg16_bn_features[i])
        self.block_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
        )
        initialize_module.initialize(modules=self.block_5.modules())
        if freeze:
            for param in self.block_1.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> namedtuple:
        '''
            Input params: `x` - a tensor of shape (batch_size, channels, height, width).
            Returns: `x` - a tensor of shape (batch_size, channels, height, width).
        '''
        block_1_out = self.block_1(x)
        block_2_out = self.block_2(block_1_out)
        block_3_out = self.block_3(block_2_out)
        block_4_out = self.block_4(block_3_out)
        block_5_out = self.block_5(block_4_out)
        vgg16_bn_output = namedtuple('VGG16_BN_OUTPUT', ['block_1_out', 'block_2_out', 'block_3_out', 'block_4_out', 'block_5_out'])
        return vgg16_bn_output(block_1_out, block_2_out, block_3_out, block_4_out, block_5_out)


if __name__ == '__main__':
    vgg16_bn_model = VGG16_BN()
    output = vgg16_bn_model(torch.randn(1, 3, 224, 224))
    print("VGG16_BN block 1 output: ", output.block_1_out.shape)
    print("VGG16_BN block 2 output: ", output.block_2_out.shape)
    print("VGG16_BN block 3 output: ", output.block_3_out.shape)
    print("VGG16_BN block 4 output: ", output.block_4_out.shape)
    print("VGG16_BN block 5 output: ", output.block_5_out.shape)
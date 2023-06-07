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
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for i in range(13):
            self.slice1.add_module(f"{str(i)}", vgg16_bn_features[i])
        for i in range(13, 20):
            self.slice2.add_module(f"{str(i)}", vgg16_bn_features[i])
        for i in range(20, 30):
            self.slice3.add_module(f"{str(i)}", vgg16_bn_features[i])
        for i in range(30, 40):
            self.slice4.add_module(f"{str(i)}", vgg16_bn_features[i])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
        )
        initialize_module.initialize(modules=self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> namedtuple:
        '''
            Input params: `x` - a tensor of shape (batch_size, channels, height, width).
            Returns: `x` - a tensor of shape (batch_size, channels, height, width).
        '''
        slice1_out = self.slice1(x)
        slice2_out = self.slice2(slice1_out)
        slice3_out = self.slice3(slice2_out)
        slice4_out = self.slice4(slice3_out)
        slice5_out = self.slice5(slice4_out)
        vgg16_bn_output = namedtuple('VGG16_BN_OUTPUT', ['slice1_out', 'slice2_out', 'slice3_out', 'slice4_out', 'slice5_out'])
        return vgg16_bn_output(slice1_out, slice2_out, slice3_out, slice4_out, slice5_out)


if __name__ == '__main__':
    vgg16_bn_model = VGG16_BN()
    output = vgg16_bn_model(torch.randn(1, 3, 224, 224))
    print("VGG16_BN slice 1 output: ", output.slice1_out.shape)
    print("VGG16_BN slice 2 output: ", output.slice2_out.shape)
    print("VGG16_BN slice 3 output: ", output.slice3_out.shape)
    print("VGG16_BN slice 4 output: ", output.slice4_out.shape)
    print("VGG16_BN slice 5 output: ", output.slice5_out.shape)
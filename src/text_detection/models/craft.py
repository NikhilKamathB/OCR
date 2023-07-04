import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model.vgg16_bn import VGG16_BN
from .base_model.utils import InitializeModule, DoubleConv2d


class CRAFT(nn.Module):

    '''
        CRAFT model.
    '''

    def __init__(self, freeze: bool = False) -> None:
        '''
            Initial defintions for the CRAFT model.
            Input params: `freeze` - a boolean value to freeze the model.
            Returns: `None`.
        '''
        super().__init__()
        initialize_module = InitializeModule()
        self.basenet = VGG16_BN(freeze=freeze)
        self.upconv1 = DoubleConv2d(in_channels=1024, mid_channels=512, out_channels=256)
        self.upconv2 = DoubleConv2d(in_channels=512, mid_channels=256, out_channels=128)
        self.upconv3 = DoubleConv2d(in_channels=256, mid_channels=128, out_channels=64)
        self.upconv4 = DoubleConv2d(in_channels=128, mid_channels=64, out_channels=32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1)
        )
        initialize_module.initialize(modules=self.upconv1.modules())
        initialize_module.initialize(modules=self.upconv2.modules())
        initialize_module.initialize(modules=self.upconv3.modules())
        initialize_module.initialize(modules=self.upconv4.modules())
        initialize_module.initialize(modules=self.conv_cls.modules())
    
    def forward(self, x: torch.Tensor) -> tuple:
        '''
            Input params: `x` - a tensor of shape (batch_size, channels, height, width).
            Returns: `x` - a tensor of shape (batch_size, channels, height, width).
        '''
        slice_1_out, slice_2_out, slice_3_out, slice_4_out, slice_5_out = self.basenet(x)
        out = self.upconv1(torch.cat((slice_5_out, slice_4_out), dim=1))
        out = F.interpolate(out, size=slice_3_out.size()[2:], mode='bilinear', align_corners=False)
        out = self.upconv2(torch.cat((out, slice_3_out), dim=1))
        out = F.interpolate(out, size=slice_2_out.size()[2:], mode='bilinear', align_corners=False)
        out = self.upconv3(torch.cat((out, slice_2_out), dim=1))
        out = F.interpolate(out, size=slice_1_out.size()[2:], mode='bilinear', align_corners=False)
        feature = self.upconv4(torch.cat((out, slice_1_out), dim=1))
        out = self.conv_cls(feature)
        return (out.permute(0, 2, 3, 1), feature)


if __name__ == "__main__":
    craft_model = CRAFT()
    out, feature = craft_model(torch.randn(1, 3, 768, 768))
    print(f"`out` tensor shape containing region score and affinity score: {out.shape}")
    print(f"`feature` tensor shape representing upconv part: {feature.shape}")
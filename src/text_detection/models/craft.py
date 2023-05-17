import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model.vgg16_bn import VGG16_BN
from base_model.utils import InitializeModule, DoubleConv2d


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
        self.upconv2d_1 = DoubleConv2d(in_channels=1024, mid_channels=512, out_channels=256)
        self.upconv2d_2 = DoubleConv2d(in_channels=512, mid_channels=256, out_channels=128)
        self.upconv2d_3 = DoubleConv2d(in_channels=256, mid_channels=128, out_channels=64)
        self.upconv2d_4 = DoubleConv2d(in_channels=128, mid_channels=64, out_channels=32)
        self.conv2d_final = nn.Sequential(
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
        initialize_module.initialize(modules=self.upconv2d_1.modules())
        initialize_module.initialize(modules=self.upconv2d_2.modules())
        initialize_module.initialize(modules=self.upconv2d_3.modules())
        initialize_module.initialize(modules=self.upconv2d_4.modules())
        initialize_module.initialize(modules=self.conv2d_final.modules())
    
    def forward(self, x: torch.Tensor) -> tuple:
        '''
            Input params: `x` - a tensor of shape (batch_size, channels, height, width).
            Returns: `x` - a tensor of shape (batch_size, channels, height, width).
        '''
        block_1_out, block_2_out, block_3_out, block_4_out, block_5_out = self.basenet(x)
        out = self.upconv2d_1(torch.cat((block_5_out, block_4_out), dim=1))
        out = F.interpolate(out, size=block_3_out.size()[2:], mode='bilinear', align_corners=False)
        out = self.upconv2d_2(torch.cat((out, block_3_out), dim=1))
        out = F.interpolate(out, size=block_2_out.size()[2:], mode='bilinear', align_corners=False)
        out = self.upconv2d_3(torch.cat((out, block_2_out), dim=1))
        out = F.interpolate(out, size=block_1_out.size()[2:], mode='bilinear', align_corners=False)
        feature = self.upconv2d_4(torch.cat((out, block_1_out), dim=1))
        out = self.conv2d_final(feature)
        return (out.permute(0, 2, 3, 1), feature)


if __name__ == "__main__":
    craft_model = CRAFT()
    out, feature = craft_model(torch.randn(1, 3, 768, 768))
    print(out.shape)
    print(feature.shape)
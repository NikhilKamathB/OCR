import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class BLOCK_CNN(nn.Module):
    
    def __init__(self, in_nc, out_nc, kernel_size=3, padding=1, stride=1, pool_kernel_size=2, pool_stride=2, use_bn=False, use_max_pool=True, use_leaky_relu=True):
        super().__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.use_bn = use_bn
        self.use_max_pool = use_max_pool
        self.use_leaky_relu = use_leaky_relu
        self.conv = nn.Conv2d(self.in_nc, self.out_nc, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(self.out_nc)
        self.max_pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_max_pool:
            x = self.max_pool(x)
        if self.use_leaky_relu:
            x = F.leaky_relu(x)
        return x


class RESNET_18:

    def __init__(self, pretrained=True, freeze=False, freeze_upto_layer=5):
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet.children())[:-3]))
        if freeze:
            for ind, child in enumerate(self.resnet18.children()):
                if ind+1 <= freeze_upto_layer:
                    for params in child.parameters():
                        params.requires_grad = False
    
    def model(self):
        return self.resnet18


class CUSTOM_VGG(nn.Module):

    def __init__(self, input_channel=3):
        super().__init__()
        # self.output_channel = [int(output_channel / 8), int(output_channel / 4),
        #                        int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.conv_net = nn.Sequential(
            # conv x 2 -> max pool x 1
            BLOCK_CNN(in_nc=3, out_nc=64, kernel_size=3, stride=1, padding=1, use_max_pool=False), # 3x50x200 -> 64x50x200
            BLOCK_CNN(in_nc=64, out_nc=64, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2, use_max_pool=True), # 64x50x200 -> 64x25x100
            # conv x 2 -> max pool x 1
            BLOCK_CNN(in_nc=64, out_nc=128, kernel_size=3, stride=1, padding=1, use_max_pool=False), # 64x25x100 -> 128x25x100
            BLOCK_CNN(in_nc=128, out_nc=128, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2, use_max_pool=True), # 128x25x100 -> 128x12x50
            # conv x 2 -> max pool x 1
            BLOCK_CNN(in_nc=128, out_nc=256, kernel_size=3, stride=1, padding=1, use_max_pool=False), # 128x12x50-> 256x12x50
            BLOCK_CNN(in_nc=256, out_nc=256, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2, use_max_pool=True), # 256x12x50 -> 256x6x25
            # conv x 3 -> bn_2d x 2 -> max pool x 1
            BLOCK_CNN(in_nc=256, out_nc=1024, kernel_size=3, stride=1, padding=1, use_max_pool=False), # 256x6x25 -> 1024x6x24
            BLOCK_CNN(in_nc=1024, out_nc=1024, kernel_size=3, stride=1, padding=1, use_max_pool=False, use_bn=True), # 1024x6x25 -> 1024x6x25
            BLOCK_CNN(in_nc=1024, out_nc=1024, kernel_size=3, stride=1, padding=1, pool_kernel_size=(2, 1), pool_stride=(2, 1), use_max_pool=True, use_bn=True), # 1024x6x25 -> 1024x3x25
            # conv x 1 -> bn_2d x 1 -> max pool x 1
            BLOCK_CNN(in_nc=1024, out_nc=1024, kernel_size=3, stride=1, padding=1, pool_kernel_size=(2, 1), pool_stride=(2, 1), use_max_pool=True, use_bn=True), # 1024x3x25 -> 1024x1x25
        )
        self.conv_net.apply(weights_init)

    def forward(self, x):
        return self.conv_net(x)


class BIDIRECTIONAL_DECODER(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.gru.flatten_parameters()
        gru, _ = self.gru(x)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(gru)  # batch_size x T x output_size
        return output


class CRNN(nn.Module):

    def __init__(self, pretrained=True, freeze=False, input_channel=3, output_channel=256, hidden_size=256, vocab_size=37, default_encoder='resnet_18'):
        super().__init__()
        if default_encoder.lower() == 'custom_vgg':
            self.encoder = nn.Sequential(
                CUSTOM_VGG(input_channel=input_channel)
            )
            self.linear_head = nn.Linear(1024, 256)
        else:
            self.encoder = nn.Sequential(
                RESNET_18(pretrained=pretrained, freeze=freeze).model(),
                BLOCK_CNN(256, 256, kernel_size=3, padding=1, use_max_pool=False)
            )
            self.linear_head = nn.Linear(1024, 256)
        self.encoder_head_output = output_channel
        self.decoder = nn.Sequential(
            BIDIRECTIONAL_DECODER(self.encoder_head_output, hidden_size, hidden_size),
            BIDIRECTIONAL_DECODER(hidden_size, hidden_size, hidden_size)
        )
        self.decoder_head_ouput = hidden_size
        self.prediction = nn.Linear(self.decoder_head_ouput, vocab_size)
    
    def forward(self, input):
        visual_feature = self.encoder(input)
        visual_feature = visual_feature.permute(0, 3, 1, 2)  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.view(visual_feature.size(0), visual_feature.size(1), -1)
        visual_feature_head = F.relu(self.linear_head(visual_feature))
        # print(visual_feature)
        contextual_feature = self.decoder(visual_feature_head)
        prediction = self.prediction(contextual_feature.contiguous())
        prediction = prediction.permute(1, 0, 2)
        prediction_logits = prediction.log_softmax(2)
        # print(prediction_logits)
        return prediction, prediction_logits


if __name__ == '__main__':
    model = CRNN(pretrained=True, freeze=False, output_channel=256, hidden_size=256, vocab_size=37, default_encoder='custom_vgg')
    output = model(torch.randn(1, 3, 50, 200))
    print(output[0])
    print(output[0].shape)
    print(output[1])
    print(output[1].shape)
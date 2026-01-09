import torch
import torch.nn as nn

try:
    from ..components.base import init_weights
except ImportError:
    from implementations.networks.torch.components.base import init_weights


class Block(nn.Module):
    """
    Basic Block for ResNet18 and ResNet34
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        
        # Conv1 handles the stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        # Conv2 always has stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck Block for ResNet50, ResNet101, etc.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x
        
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
            
        x += identity
        x = self.relu(x)
        
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes=1000, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # Initial Stem
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layers
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        # Arbitrary Input Support
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        
        # Determine if we need to downsample the identity connection
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, 
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )
            
        layers = []
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        
        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    

    def reset_parameters(self):
        self.apply(init_weights)


    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
    
if __name__ == "__main__":
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=64, num_channels=3)
    model.reset_parameters()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 64)
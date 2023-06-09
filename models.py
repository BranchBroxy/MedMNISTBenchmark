import torch
import torch.nn as nn
import sparselinear as sl
class SparseResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_features, downsample=None):
        super(SparseResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Flatten(),
            sl.SparseLinear(in_features=in_channels, out_features=hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.ReLU())
        #self.flatten = nn.Flatten()
        #self.sparse = sl.SparseLinear(in_features=in_channels, out_features=hidden_features)
        #self.batch_norm = nn.BatchNorm1d(hidden_features)
        #self.relu = nn.ReLU()

        self.conv2 = nn.Sequential(
            sl.SparseLinear(in_features=hidden_features, out_features=out_channels),
            nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        """x = self.flatten(x)
        x = self.sparse(x)
        x = self.batch_norm(x)
        out = self.relu(x)"""
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SparseResNet(nn.Module):
    def __init__(self, in_features, out_features, block, layers, hidden_features=10):
        super(SparseResNet, self).__init__()
        self.inplanes = 128
        self.sparse1 = nn.Sequential(
            nn.Flatten(),
            sl.SparseLinear(in_features=in_features, out_features=self.inplanes),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU())
        # self.flatten = nn.Flatten()
        # self.sparse_in = sl.SparseLinear(in_features=in_features, out_features=hidden_features)
        # self.batch_norm = nn.BatchNorm1d(hidden_features)
        # self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block=block, in_features=64, out_features=128, blocks=layers[0], hidden_features=hidden_features)
        self.layer1 = self._make_layer(block=block, in_features=128, out_features=256, blocks=layers[1], hidden_features=hidden_features)
        self.layer2 = self._make_layer(block=block, in_features=256, out_features=512, blocks=layers[2], hidden_features=hidden_features)
        self.layer3 = self._make_layer(block=block, in_features=512, out_features=512, blocks=layers[3], hidden_features=hidden_features)
        self.avgpool = nn.AvgPool1d(8, stride=2)

        in_features = int(((512 - 8)/2)+1)
        self.fc = sl.SparseLinear(in_features, out_features)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, in_features, out_features, blocks, hidden_features):
        downsample = None
        if in_features != out_features or self.inplanes != in_features:
            downsample = nn.Sequential(
                sl.SparseLinear(in_features=in_features, out_features=out_features),
                nn.BatchNorm1d(out_features),
            )
        layers = []
        layers.append(block(in_features, out_features, hidden_features, downsample=downsample))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(out_features, out_features, hidden_features, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.flatten(x)
        #x = self.sparse_in(x)
        #x =self.batch_norm(x)
        #x=self.relu(x)
        x = self.sparse1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

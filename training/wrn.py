import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.bn1 = norm_layer(in_planes, momentum=0.9)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = norm_layer(out_planes, momentum=0.9)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth, width_factor, in_channels, dropout_rate=0.0, label_dim=10, use_bn=True):
        super().__init__()
        self.depth = depth
        self.width_factor = width_factor
        self.n_channels = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]
        self.dropout_rate = dropout_rate
        norm_layer = nn.BatchNorm2d if use_bn else Identity

        assert (depth - 4) % 6 == 0, "Wide-ResNet depth should be 6n+4"
        n = (depth - 4) // 6

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(n, self.n_channels[0], self.n_channels[1], 1, norm_layer)
        self.block2 = self._make_layer(n, self.n_channels[1], self.n_channels[2], 2, norm_layer)
        self.block3 = self._make_layer(n, self.n_channels[2], self.n_channels[3], 2, norm_layer)

        self.bn1 = norm_layer(self.n_channels[3], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_channels[3], label_dim)

        self._init_weights()

    def _make_layer(self, n, in_planes, out_planes, stride, norm_layer):
        layers = []
        for i in range(n):
            layers.append(BasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, self.dropout_rate, norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avg_pool(out)
        out = out.view(-1, self.n_channels[3])
        return self.fc(out)

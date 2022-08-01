import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super(ResBlock, self).__init__()

        self.branchA = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                stride=stride,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=out_c
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(
                num_features=out_c
            ),
        )
        self.relu = nn.ReLU()
        self.identity_downsample = None
        if stride != 1 or in_c != out_c:
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.branchA(x)
        if self.identity_downsample != None:
            identity = self.identity_downsample(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.Network = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64, kernel_size=7, stride=2
            ),
            nn.BatchNorm2d(
                num_features=64
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3, stride=2
            ),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),

            ResBlock(128, 256, 2),
            nn.Dropout(p=0.70),
            ResBlock(256, 512, 2),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.Network(x)



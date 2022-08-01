from torch import nn

class ConnectionBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.pathA = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                stride=stride,
                padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathB = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=5,
                stride=stride,
                padding=2),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathC = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=9,
                stride=stride,
                padding=4),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=9,
                stride=1,
                padding=4),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathSkip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=stride),
            nn.BatchNorm2d(num_features=out_c))


    def forward(self, x):
        return self.pathA(x) + self.pathB(x) + self.pathC(x) + self.pathSkip(x)


class FunNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConnectionBlock(in_c=64, out_c=64, stride=1),
            ConnectionBlock(in_c=64, out_c=128, stride=2),
            ConnectionBlock(in_c=128, out_c=256, stride=2),
            ConnectionBlock(in_c=256, out_c=512, stride=2),
            nn.AvgPool2d(kernel_size=(10, 10)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=2),
            nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
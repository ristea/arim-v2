import torch.nn as nn


class FConv3Net(nn.Module):
    def __init__(self):
        super(FConv3Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=13, stride=1, padding=(12, 0), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=13, stride=1, padding=(12, 0), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=13, stride=1, padding=(12, 0), padding_mode='circular')

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=9, stride=1, padding=(8, 0), padding_mode='circular')
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=(8, 0), padding_mode='circular')
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=9, stride=1, padding=(8, 0), padding_mode='circular')

        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 1))

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')

        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv9 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=5, stride=1, padding=(4, 0), padding_mode='circular')
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1)

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)

        x = self.pool1(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.conv6(x)
        x = self.activation(x)

        x = self.pool2(x)

        x = self.conv7(x)
        x = self.activation(x)
        x = self.conv8(x)
        x = self.activation(x)

        x = self.pool3(x)

        x = self.conv9(x)
        x = self.conv10(x)

        return x[:, :, :, 0]

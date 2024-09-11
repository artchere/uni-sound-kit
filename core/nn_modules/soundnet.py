import torch
from torch import nn


class SoundNet(nn.Module):
    """
    Tensor shape: [Batch x Channel (1) x Time] or [Batch x Time]
    """
    def __init__(self):
        """
        For 1D raw data
        """
        super(SoundNet, self).__init__()

        self.conv1 = nn.LazyConv1d(16, kernel_size=64, stride=2)
        self.batchnorm1 = nn.BatchNorm1d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(8, 8)

        self.conv2 = nn.LazyConv1d(32, kernel_size=32, stride=2)
        self.batchnorm2 = nn.BatchNorm1d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(8, 2)

        self.conv3 = nn.LazyConv1d(64, kernel_size=16, stride=2)
        self.batchnorm3 = nn.BatchNorm1d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.1)

        self.conv4 = nn.LazyConv1d(128, kernel_size=8, stride=2)
        self.batchnorm4 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.1)

        self.conv5 = nn.LazyConv1d(256, kernel_size=4, stride=2)
        self.batchnorm5 = nn.BatchNorm1d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool1d(4, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((128, 1))

    def forward(self, waveform):
        assert 2 <= len(waveform.shape) <= 3, "Incompatible tensor shape"

        # BxT -> BxCxT
        waveform = waveform.unsqueeze(1) if len(waveform.shape) == 2 else waveform

        x = self.conv1(waveform)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.avgpool(x)

        return x.squeeze(-1)


if __name__ == "__main__":
    # Test array like 1-channel audio with 5 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 80000)

    # Construct the final model:
    model = SoundNet()

    print(model(test_tensor).shape)

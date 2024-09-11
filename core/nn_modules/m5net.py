import torch
from torch import nn


class M5Net(nn.Module):
    def __init__(
            self,
            sr: int = 16000,
            n_channels: int = 32,
            stride: int = 16
    ):
        super().__init__()
        self.conv1 = nn.LazyConv1d(n_channels, kernel_size=sr//100, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(n_channels, 2 * n_channels, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channels)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channels, 2 * n_channels, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channels)
        self.pool4 = nn.MaxPool1d(4)

    def forward(self, waveform):
        assert 2 <= len(waveform.shape) <= 3, "Incompatible tensor shape"

        # BxT -> BxCxT
        waveform = waveform.unsqueeze(1) if len(waveform.shape) == 2 else waveform

        x = self.conv1(waveform)
        x = nn.ReLU()(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = nn.ReLU()(self.bn4(x))
        x = self.pool4(x)
        return x


if __name__ == "__main__":
    # Test array like 1-channel audio with 4 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 64000)

    # Construct the final model:
    model = M5Net(sr=16000, n_channels=128, stride=8)

    print(model(test_tensor).shape)

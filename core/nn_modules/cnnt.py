import torch
import torch.nn as nn


class CNNT(nn.Module):
    def __init__(self, freq_size: int = None, out_dim: int = None):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            # 2. conv block
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.25),
            # 3. conv block
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=8),
            nn.Dropout(p=0.25)
        )
        # Transformer block
        self.transf_maxpool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        transf_layer = nn.TransformerEncoderLayer(
            batch_first=False, d_model=freq_size//2, nhead=4, dim_feedforward=256, dropout=0.4, activation='relu'
        )
        self.transf_encoder = nn.TransformerEncoder(transf_layer, num_layers=4)
        self.fc = nn.LazyLinear(out_dim)

    def forward(self, x):
        assert 3 <= len(x.shape) <= 4, "Incorrect shape of input data."

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        conv_embedding = self.conv2Dblock(x)  # (B ,C, F, T)
        conv_embedding = torch.flatten(conv_embedding, start_dim=1)

        x_reduced = self.transf_maxpool(x)
        x_reduced = torch.squeeze(x_reduced, 1)
        x_reduced = x_reduced.permute(2, 0, 1)  # requires shape = (T, B, embedding)
        transf_out = self.transf_encoder(x_reduced)
        transf_embedding = torch.mean(transf_out, dim=0)

        complete_embedding = torch.cat([conv_embedding, transf_embedding], dim=1)

        return self.fc(complete_embedding)


if __name__ == "__main__":
    model = CNNT(80)

    print(model(torch.randn(2, 80, 300)).shape)

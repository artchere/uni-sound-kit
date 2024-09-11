import torch
import torch.nn as nn

from core.nn_modules.extractor import FeatureExtractor

# All credits to: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346


class TimeDistributed(nn.Module):
    """
    Mimics the Keras TimeDistributed layer.
    """
    def __init__(self, module: torch.nn.Module, batch_first: bool, layer_name: str):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.layer_name = layer_name

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-3), x.size(-2), x.size(-1))

        y = self.module(x_reshape)

        if self.layer_name == "convolutional" or self.layer_name == "max_pooling":
            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(
                    x.size(0), x.size(1), y.size(-3), y.size(-2), y.size(-1)
                )
            else:
                y = y.view(-1, x.size(1), y.size(-1))

        else:
            # We have to reshape Y
            if self.batch_first:
                y = y.contiguous().view(x.size(0), x.size(1), y.size(-1))
            else:
                y = y.view(-1, x.size(1), y.size(-1))

        return y


def weight_init(m: torch.nn.Module):
    """
    Initalize all the weights in the PyTorch model to be the same as Keras.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)


class ExtractLSTMOutput(nn.Module):
    """
    Extracts only the output from the BiLSTM layer.
    """
    def forward(self, x):
        output, _ = x
        return output


class VoxSeg(nn.Module):
    """
    Creates the Voxseg model in PyTorch.
    Tensor shape: [Batch x 1 x Frequency x Time]
    https://github.com/rafaelgreca/voxseg-pytorch/tree/main
    """
    def __init__(self):
        super(VoxSeg, self).__init__()
        self.layers = nn.Sequential(
            TimeDistributed(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
                batch_first=True,
                layer_name="convolutional",
            ),
            nn.ELU(),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=2), batch_first=True, layer_name="max_pooling"
            ),
            TimeDistributed(nn.Flatten(), batch_first=True, layer_name="flatten"),
            TimeDistributed(
                nn.LazyLinear(out_features=128),
                batch_first=True,
                layer_name="dense",
            ),
            nn.Dropout(p=0.25),
            nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            ),
            ExtractLSTMOutput()
        )

    def forward(self, x):
        assert 3 <= len(x.shape) <= 4, "Incorrect shape of input data."

        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        return self.layers(x)


if __name__ == "__main__":
    extractor = FeatureExtractor(
        extractor_type="melspec",
        use_delta_order=0,
        add_channel_dim=False,
        target_sr=16000,
        win_length=0.040,
        hop_length=0.010
    )

    # Test array like 1-channel audio with 2 sec length (batch size == 2):
    test_tensor = torch.randn(2, 1, 32000)

    print(extractor(test_tensor).shape)

    # Construct the final model:
    model = nn.Sequential(
        extractor,
        VoxSeg()
    )

    print(model(test_tensor).shape)

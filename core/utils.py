import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class PDDataset(Dataset):
    """
    Single data batch generator with base audio preprocessing
    """
    def __init__(self, data: tuple = None, target_sr: int = None, target_length: int = None):
        self.filepaths, self.targets = data
        self.target_sr = target_sr
        self.target_length = target_length
        self.dataprocessor = DataProcessor(target_sr=self.target_sr, target_length=self.target_length)

    def __len__(self):
        return len(self.filepaths)

    def load_audio(self, audio_path):
        return self.dataprocessor.read_file(audio_path, train_mode=True)

    def __getitem__(self, idx):
        audio_filepath = self.filepaths[idx]
        label = self.targets[idx]
        waveform = self.load_audio(audio_filepath)

        return waveform, torch.tensor(label, dtype=torch.long)


class DataProcessor:
    """
    Standart ops such reading, resampling and cutting/padding
    """
    def __init__(self, target_sr: int = 16000, target_length: int = 8):
        self.target_sr = target_sr
        self.target_length = int(target_sr * target_length)

    def read_file(
        self,
        filepath: str = None,
        train_mode: bool = False
    ):
        # Load to single-channel with channel averaging
        samples, sr = torchaudio.load(filepath)
        samples = torch.mean(samples, dim=0) if samples.shape[0] > 1 else samples

        # Resampling and reshaping
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
        samples = resampler(samples).squeeze() if sr != self.target_sr else samples.squeeze()

        # Simple random pos and pad/cut ops for train mode
        samples_length = int(len(samples))
        random_pos_coef = np.random.choice([0.0, 0.125, 0.25], replace=False)

        if samples_length < self.target_length:
            diff = self.target_length - samples_length
            pad = torch.zeros(diff)
            if train_mode and diff > 99:
                left = int(random_pos_coef * diff)
                right = diff - left
                samples = torch.hstack((pad[:left], samples, pad[:right]))
            else:
                samples = torch.hstack((pad, samples))

        if samples_length > self.target_length:
            diff = samples_length - self.target_length
            left = diff // 2
            right = diff - left
            if train_mode and diff > 99:
                samples = samples[left:samples_length-right]
            else:
                samples = samples[:samples_length-diff]

        return samples.unsqueeze(0)


def plot_confusion_matrix(y_pred, y_true, labels, display_labels, save_path):
    """
    Plots the confusion matrix for given data
    :param save_path: Path to history directory
    :param y_pred: Predicted targets
    :param y_true: True targets
    :param labels: Class labels integer
    :param display_labels: Class labels to display
    """
    cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(
        cmap=plt.cm.Blues, values_format="d"
    )

    save_path = os.path.normpath(save_path.replace('\\', '/'))

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_history(history_dict: dict = None, save_path: str = None):
    x_epochs = []
    y_train_losses = []
    y_val_losses = []
    y_val_precision = []
    y_val_recall = []
    y_val_f1 = []

    for epoch, values in history_dict.items():
        train_loss, val_loss, precision, recall, f1 = values
        x_epochs += [int(epoch)]
        y_train_losses += [train_loss]
        y_val_losses += [val_loss]
        y_val_precision += [precision]
        y_val_recall += [recall]
        y_val_f1 += [f1]

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 14))

    ax[0].plot(x_epochs, y_train_losses, label="Train Loss")
    ax[0].plot(x_epochs, y_val_losses, label="Val Loss")
    ax[1].plot(x_epochs, y_val_precision, label="Val Precision")
    ax[1].plot(x_epochs, y_val_recall, label="Val Recall")
    ax[1].plot(x_epochs, y_val_f1, label="Val F1")

    ax[0].set_xlabel("Epochs", fontsize=14)
    ax[0].set_ylabel("Loss", fontsize=14)
    ax[0].legend(loc='upper left', fontsize=8)
    ax[1].set_xlabel("Epochs", fontsize=14)
    ax[1].set_ylabel("Metrics", fontsize=14)
    ax[1].legend(loc='upper left', fontsize=8)

    fig.suptitle(f"Train vs Val", fontsize=18)
    fig.subplots_adjust(top=0.92)

    save_path = os.path.normpath(save_path.replace('\\', '/'))

    fig.savefig(save_path)
    plt.close(fig)


def test_memory_requirement(model: torch.nn.Module, test_tensor: torch.Tensor, device: str):
    from torch.profiler import profile, ProfilerActivity

    model.to(device)
    test_tensor = test_tensor.to(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        model(test_tensor)
    stats = prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model(test_tensor)
        memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        return f"FF memory usage on {test_tensor.shape} tensor: {memory_usage / 1024 ** 2} MB\n" \
               f"{stats}"
    else:
        return f"CPU and memory usage stats on {test_tensor.shape} tensor:\n" \
               f"{stats}"

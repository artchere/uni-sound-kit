import torch
import torchaudio
from torch import nn

import librosa
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            extractor_type: str = 'mfcc',
            use_delta_order: int = 0,
            add_channel_dim: bool = False,
            target_sr: int = 16000,
            preemphasis: bool = True,
            win_length: float = 0.032,
            hop_length: float = 0.008,
            n_coefs: int = 40,
            n_filters: int = 80
                 ):
        super(FeatureExtractor, self).__init__()
        self.extractor_type = extractor_type
        self.use_delta_order = use_delta_order
        self.add_channel_dim = add_channel_dim
        self.preemphasis = preemphasis
        self.target_sr = target_sr
        self.n_fft = int(round(win_length * target_sr, 0))
        self.hop_length = int(round(hop_length * target_sr, 0))

        self.stft_transformer = nn.Sequential(
            torchaudio.transforms.Spectrogram(n_fft=self.n_fft,hop_length=self.hop_length),
            torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        )
        self.melspectrogram_transformer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sr,
                n_mels=n_filters,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                f_min=50
            ),
            torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        )
        self.mfcc_transformer = torchaudio.transforms.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_coefs,
            melkwargs={
                "n_mels": n_filters,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "f_min": 50
            }
        )

    @staticmethod
    def get_deltas(features=None, order=None):
        # [BxFxT] (1-я производная: скорость):
        delta = torchaudio.functional.compute_deltas(features)

        if order == 2:
            # [BxFxT] (2-я производная: ускорение):
            delta = torchaudio.functional.compute_deltas(delta)

        return delta

    def get_melspectrum(self, samples=None):
        melspec = self.melspectrogram_transformer(samples)
        melspec -= torch.mean(melspec, dim=-1, keepdim=True)

        return melspec

    def get_mfcc(self, samples=None):
        mfccs = self.mfcc_transformer(samples)
        mfccs -= torch.mean(mfccs, dim=-1, keepdim=True)

        return mfccs

    def get_spectral_features(self, samples=None):
        # Матрица резонансных (фундаментальных) частот
        f0 = np.expand_dims(
            librosa.yin(
                y=samples.cpu().numpy(),
                sr=self.target_sr,
                fmin=50,
                fmax=int(self.target_sr / 2),
                frame_length=self.n_fft - 1,
                hop_length=self.hop_length
            ),
            axis=1
        )

        # Точки пересечения нуля
        zcr = librosa.feature.zero_crossing_rate(
            samples.cpu().numpy(),
            frame_length=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Среднеквадратичное значение энергии для каждого кадра
        rms = librosa.feature.rms(
            y=samples.cpu().numpy(),
            frame_length=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Спектральный центроид (частотный центр масс звука)
        spec_centroid = librosa.feature.spectral_centroid(
            y=samples.cpu().numpy(),
            sr=self.target_sr,
            n_fft=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Спектральный контраст (высокие значения == чёткие узкополосные сигналы, низкие == широкополосные шумы)
        spec_contrast = librosa.feature.spectral_contrast(
            y=samples.cpu().numpy(),
            sr=self.target_sr,
            fmin=50,
            n_fft=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Спектральная мера плоскостности (тон / не тон)
        spec_flatness = librosa.feature.spectral_flatness(
            y=samples.cpu().numpy(),
            n_fft=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Спектральный спад (показатель спада ВЧ более чем на ~85% по каждому частотному фильтру)
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=samples.cpu().numpy(),
            sr=self.target_sr,
            n_fft=self.n_fft - 1,
            hop_length=self.hop_length
        )

        # Stack:
        result = np.concatenate(
            (
                f0,
                zcr,
                rms,
                spec_centroid,
                spec_contrast,
                spec_flatness,
                spec_rolloff
            ), axis=-2
        )

        return result

    def forward(self, samples: torch.Tensor):
        samples = torchaudio.functional.preemphasis(samples) if self.preemphasis else samples

        assert 2 <= len(samples.shape) <= 3, "Incorrect shape of input data."

        if self.extractor_type == "melspec":
            features = self.get_melspectrum(samples)
        elif self.extractor_type == "mfcc":
            features = self.get_mfcc(samples)
        elif self.extractor_type == "spectral":  # TODO: GPU support
            features = torch.from_numpy(self.get_spectral_features(samples)).float()
        else:
            raise ValueError("Invalid extractor type")

        features = features if self.use_delta_order == 0 else self.get_deltas(features, self.use_delta_order)

        if self.add_channel_dim:
            features = features.unsqueeze(1) if len(features.shape) == 3 else features  # BxFxT -> BxCxFxT
        elif self.use_delta_order != 3:
            features = features.squeeze(1) if len(features.shape) == 4 else features  # BxCxFxT -> BxFxT

        return features


if __name__ == "__main__":
    extractor = FeatureExtractor(extractor_type="spectral",
                                 use_delta_order=0,
                                 add_channel_dim=False,
                                 target_sr=16000,
                                 win_length=0.032,
                                 hop_length=0.032)

    # Test array like 1-channel audio with 1 sec length at 16 kHz SR (batch size == 2):
    test_tensor = torch.randn(2, 1, 16000)

    print(extractor(test_tensor).shape)

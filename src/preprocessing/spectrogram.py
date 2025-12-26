import torch
from torch import nn
import torchaudio

class SpectrogramGetter(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64):
        super().__init__()
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160
        )

    def forward(self, audio):
        # audio: (Batch, Time) или (1, Time)
        spec = self.spec(audio) # (Batch, Mels, Time')
        # Берем логарифм, чтобы значения были стабильнее для нейросети
        return torch.log(spec + 1e-9)

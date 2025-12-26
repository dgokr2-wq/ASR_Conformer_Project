import torch
from torch import nn

class ConformerModel(nn.Module):
    def __init__(self, n_feats=64, n_tokens=28, d_model=256):
        super().__init__()
        # Сжимаем время в 4 раза (Subsampling)
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.proj = nn.Linear(d_model * (n_feats // 4), d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, **batch):
        x = spectrogram.unsqueeze(1) # (B, 1, F, T)
        x = self.subsample(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.proj(x)
        x = self.encoder(x)
        return {"logits": self.fc(x)}

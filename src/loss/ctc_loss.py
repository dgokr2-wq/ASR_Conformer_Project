import torch
from torch import nn
import torch.nn.functional as F

class CTCLossWrapper(nn.Module):
    def __init__(self, blank_idx=0):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, logits, text, logits_length, text_encoding_length, **batch):
        # CTC ждет (T, B, V)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        loss = self.loss_fn(log_probs, text, logits_length, text_encoding_length)
        return {"loss": loss}

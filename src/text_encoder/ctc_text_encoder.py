import torch
import numpy as np

class CTCTextEncoder:
    def __init__(self):
        self.alphabet = " abcdefghijklmnopqrstuvwxyz'"
        self.vocab = ["<blank>"] + list(self.alphabet)
        self.char2idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx2char = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text):
        return torch.Tensor([self.char2idx[c] for c in text.lower() if c in self.char2idx]).long()

    def ctc_decode(self, inds):
        res = []
        last = None
        for i in inds:
            if i != last and i != 0:
                res.append(self.idx2char[i])
            last = i
        return "".join(res)

    def beam_search(self, logits, beam_size=5):
        # logits: (T, V)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dp = {("", "<blank>"): 1.0}
        for t in range(probs.shape[0]):
            new_dp = {}
            for i, p in enumerate(probs[t]):
                char = self.idx2char[i]
                for (prefix, last), prob in dp.items():
                    new_pref = prefix if char == last or char == "<blank>" else prefix + char
                    new_dp[(new_pref, char)] = new_dp.get((new_pref, char), 0) + prob * p.item()
            dp = dict(sorted(new_dp.items(), key=lambda x: x[1], reverse=True)[:beam_size])
        return sorted(dp.items(), key=lambda x: x[1], reverse=True)[0][0][0]

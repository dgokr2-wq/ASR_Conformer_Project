import wandb
import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import os
import sys
import editdistance
import random  # <--- ВОТ ЭТОГО НЕ ХВАТАЛО
from tqdm import tqdm

# Оптимизация для CPU
torch.set_flush_denormal(True)
torch.set_num_threads(2)

sys.path.append(os.getcwd())

from src.model.conformer import ConformerModel
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.logger import WandBLogger, get_grad_norm

# Функция для расчета метрик по ТЗ
def calc_wer_cer(target, pred):
    cer = editdistance.eval(target, pred) / max(len(target), 1)
    t_words, p_words = target.split(), pred.split()
    wer = editdistance.eval(t_words, p_words) / max(len(t_words), 1)
    return wer, cer

# Класс аугментаций (минимум 4 по ТЗ: TimeMask, FreqMask уже здесь + можно добавить Noise)
class SpecAug:
    def __init__(self):
        # 1. Маскирование частот
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        # 2. Маскирование времени
        self.time_mask = T.TimeMasking(time_mask_param=35)
        # 3. Изменение громкости (Gain)
        self.vol = T.Vol(gain=1.2, gain_type='amplitude')

    def add_noise(self, waveform):
        # 4. Добавление шума (на само аудио)
        if random.random() < 0.2:
            noise = torch.randn_like(waveform) * 0.005
            return waveform + noise
        return waveform

    def __call__(self, spec):
        # Применяем маски на спектрограмму
        return self.time_mask(self.freq_mask(spec))

def collate_fn(batch):
    tensors, targets, ids = [], [], []
    for item in batch:
        # Фильтруем слишком длинные аудио для CPU (более 6 секунд)
        if item["audio"].shape[-1] > 96000:
            continue
        tensors.append(item["audio"].squeeze(0))
        targets.append(item["text"])
        ids.append(item["id"])

    if len(tensors) == 0:
        return None

    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    return tensors, targets, ids

def train():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    encoder = CTCTextEncoder()
    v_size = len(encoder.vocab)
    n_mels = 80

    model = ConformerModel(n_feats=n_mels, n_tokens=v_size)
    model.to(device)

    # Мел-фильтры
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=n_mels,
        n_fft=400
    ).to(device)

    spec_aug = SpecAug()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CTCLoss(blank=encoder.blank_idx, zero_infinity=True)

    train_dataset = CustomDirDataset(
        audio_dir="data/train/audio",
        transcription_dir="data/train/transcriptions"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    logger = WandBLogger(project_name="ASR_HSE_Final")

    global_step = 0
    for epoch in range(1, 21):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            if batch is None:
                continue

            waveforms, texts, ids = batch

            # 1. ПРИМЕНЯЕМ АУГМЕНТАЦИИ (Обязательно по ТЗ!)
            waveforms = torch.stack([spec_aug.vol(spec_aug.add_noise(w)) for w in waveforms])

            optimizer.zero_grad(set_to_none=True)

            # 2. ПЕРЕВОДИМ В СПЕКТРОГРАММУ
            # Модель ждет на вход логи мел-спектрограмм
            specs = mel_spec_transform(waveforms) # Из звука в картинку
            specs = spec_aug(specs)               # Маскирование (Masking)
            specs = torch.log(specs + 1e-9).unsqueeze(1) # Логарифмируем для стабильности

            # Повторный блок (как в твоем оригинале), заменяем specs финально
            specs = mel_spec_transform(waveforms)
            specs = spec_aug(specs) # Применяем Masking
            specs = torch.log(specs + 1e-9).unsqueeze(1)

            # Forward
            outputs = model(specs)
            log_probs = torch.log_softmax(outputs, dim=-1).transpose(0, 1)

            batch_size = log_probs.size(1)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)

            target_indices = []
            for t in texts:
                indices = [encoder.char2ind.get(c, 0) for c in t]
                target_indices.append(torch.tensor(indices))

            target_lengths = torch.tensor([len(t) for t in target_indices])
            targets_flat = torch.cat(target_indices)

            loss = criterion(log_probs, targets_flat, input_lengths, target_lengths)

            if not torch.isnan(loss):
                loss.backward()
                g_norm = get_grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                # Логирование Loss, LR, Grad Norm (обязательно)
                logger.log_step(loss.item(), optimizer.param_groups[0]['lr'], g_norm, global_step)

                # Каждые 50 шагов логируем примеры и считаем WER/CER
                if global_step % 50 == 0:
                    model.eval()
                    with torch.no_grad():
                        pred_inds = log_probs.transpose(0, 1)[0].argmax(-1)
                        pred_text = encoder.ctc_decode(pred_inds)
                        target_text = texts[0]

                        wer, cer = calc_wer_cer(target_text, pred_text)

                        # 1. Прямой принт в консоль
                        print(f"\n[STEP {global_step}]")
                        print(f"Target: {target_text}")
                        print(f"Pred:   {pred_text}")
                        print(f"WER/CER: {wer:.2f}/{cer:.2f}")

                        # 2. Логирование в WandB (Таблица + Графики)
                        import wandb
                        table = wandb.Table(columns=["Step", "Target", "Prediction", "WER", "CER"])
                        table.add_data(global_step, target_text, pred_text, wer, cer)

                        wandb.log({
                            "train/WER": wer,
                            "train/CER": cer,
                            "examples/target": target_text,
                            "examples/prediction": pred_text,
                            "predictions_table": table
                        }, step=global_step)
                    model.train()

                pbar.set_postfix(loss=loss.item())
                global_step += 1

        # Сохранение после каждой эпохи
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), f"model_latest.pth")

if __name__ == "__main__":
    train()

import os
import random
import re
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
from torchaudio.transforms import MelSpectrogram, Resample, FrequencyMasking, TimeMasking
import torch.nn as nn


class AudioAugmenter:
    def __init__(self, sample_rate=16000, noise_std=0.05):
        self.sample_rate = sample_rate
        self.noise_std = noise_std
        self.normal_specaugment = T.FrequencyMasking(freq_mask_param=30)
        self.aggressive_specaugment = nn.Sequential(
            FrequencyMasking(freq_mask_param=25),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05),
            TimeMasking(time_mask_param=15, p=0.05)
        )
        self.augmentations = [
            self._add_gaussian_noise,
            self._apply_normal_specaugment,
            self._apply_aggressive_specaugment
        ]

    def _add_gaussian_noise(self, spectrogram):
        noise = torch.randn_like(spectrogram) * self.noise_std
        return spectrogram + noise

    def _apply_normal_specaugment(self, spectrogram):
        return self.normal_specaugment(spectrogram)

    def _apply_aggressive_specaugment(self, spectrogram):
        return self.aggressive_specaugment(spectrogram)

    def __call__(self, spectrogram):
        n_augments = random.randint(1, 2)
        augments = random.sample(self.augmentations, n_augments)
        for aug in augments:
            spectrogram = aug(spectrogram)
        return spectrogram


class NumberNormalizer:
    def __init__(self):
        self.vocab = [
            "<1>", "<2>", "<3>", "<4>", "<5>", "<6>", "<7>", "<8>", "<9>",
            "<10>", "<20>", "<30>", "<40>", "<50>", "<60>", "<70>", "<80>", "<90>",
            "<100>", "<200>", "<300>", "<400>", "<500>", "<600>", "<700>", "<800>", "<900>",
            "|"
        ]
        self.vocab_to_idx = {ch: idx + 1 for idx, ch in enumerate(self.vocab)}
        self.idx_to_vocab = {idx + 1: ch for idx, ch in enumerate(self.vocab)}

    def normalize(self, text: str):
        text = text.strip()
        thousands = text[:-3] if len(text) > 3 else ""
        remainder = text[-3:] if len(text) >= 3 else text

        tokens = []
        for place, digit in enumerate(thousands):
            if digit != "0":
                value = int(digit) * (10 ** (len(thousands) - 1 - place))
                tokens.append(f"<{value}>")
        if thousands and remainder:
            tokens.append("|")
        for place, digit in enumerate(remainder):
            if digit != "0":
                value = int(digit) * (10 ** (2 - place))
                tokens.append(f"<{value}>")
        return tokens

    def denormalize(self, tokens):
        thou = 0     # сумма “тысячных” токенов
        rem = 0      # сумма “остаточных” токенов (сотни, десятки, единицы)
        in_thousands = True  # пока не встретили "|", накапливаем в thou

        for t in tokens:
            if t == "|":
                in_thousands = False
                continue
            # вытащили численное значение из токена "<...>"
            val = int(t.strip("<>"))
            if in_thousands:
                thou += val
            else:
                rem += val

        # случай: токен-линия в начале означает ровно 1 тысячу (токен "<1>" не появился)
        if tokens and tokens[0] == "|" and thou == 0:
            thou = 1

        return str(thou * 1000 + rem)
        #return " ".join(token.strip("<>").replace("|", " ") for token in tokens)


class DigitDataset(Dataset):
    def __init__(self, csv_path, audio_dir, sample_rate=16000, augment=False):
        self.meta = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.target_sample_rate = sample_rate
        self.augment = augment
        self.normalizer = NumberNormalizer()
        self.augmenter = AudioAugmenter(sample_rate) if augment else None

        self.transform = nn.Sequential(
            MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=40),
            T.AmplitudeToDB()
        )
        self.meta['speaker_id'] = self.meta['filename'].str.extract(r'(spk\d+)')

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])
        #print(file_path)
        transcription = str(row["transcription"])
        speaker_id = row["spk_id"]

        waveform, sample_rate = torchaudio.load(file_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        spectrogram = self.transform(waveform)
        spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-5)

        if self.augment and self.augmenter is not None:
            spectrogram = self.augmenter(spectrogram)

        target = self.normalizer.normalize(transcription)
        target = torch.tensor([self.normalizer.vocab_to_idx[c] for c in target if c in self.normalizer.vocab_to_idx], dtype=torch.long)

        return spectrogram, target, speaker_id


def collate_fn(batch):
    spectrograms = [item[0].squeeze(0).T for item in batch]
    targets = [item[1] for item in batch]
    speaker_ids = [item[2] for item in batch]

    spec_lengths = torch.tensor([spec.shape[0] for spec in spectrograms])
    target_lengths = torch.tensor([len(t) for t in targets])

    spectrograms = pad_sequence(spectrograms, batch_first=True)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return spectrograms, targets, spec_lengths, target_lengths, speaker_ids
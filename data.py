import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MelSpectrogram, Resample
import random
import torchaudio.transforms as T
import re


def collate_fn(batch):
    max_len = max(spec.shape[1] for spec, _, _ in batch)
    
    spectrograms = []
    for spec, _, _ in batch:
        spectrograms.append(spec.squeeze(0).T)
    
    targets = [torch.tensor(item[1]) for item in batch]
    spec_lengths = torch.tensor([spec.shape[0] for spec in spectrograms])
    target_lengths = torch.tensor([len(t) for t in targets])
    speaker_ids = [item[2] for item in batch]
    
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return spectrograms, targets, spec_lengths, target_lengths, speaker_ids

class AudioAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_factor = 0.005
        
        self.augmentations = [
            self._add_gaussian_noise,
            #self._time_stretch,
            self._pitch_shift,
            self._add_background_noise
        ]
    
    def _add_gaussian_noise(self, waveform):
        noise = torch.randn_like(waveform) * self.noise_factor
        return waveform + noise
    
    def _time_stretch(self, waveform):
        stretch = T.TimeStretch(fixed_rate=None, n_freq=201)
        stretch_factor = random.uniform(0.8, 1.2)
        return stretch(waveform, stretch_factor)
    
    def _pitch_shift(self, waveform):
        n_steps = random.uniform(-2, 2)
        pitch_shift = T.PitchShift(self.sample_rate, n_steps)
        return pitch_shift(waveform)
    
    def _add_background_noise(self, waveform):
        noise_factor = random.uniform(0.001, 0.005)
        noise = torch.randn_like(waveform) * noise_factor
        return waveform + noise
    
    def __call__(self, waveform):
        n_augments = random.randint(1, 2)
        augments = random.sample(self.augmentations, n_augments)
        
        for aug in augments:
            waveform = aug(waveform)
        
        return waveform

class RussianNumberNormalizer:
    def __init__(self):
        self.number_mapping = {
            'ноль': '0',
            'один': '1', 'одна': '1',
            'два': '2', 'две': '2',
            'три': '3',
            'четыре': '4',
            'пять': '5',
            'шесть': '6',
            'семь': '7',
            'восемь': '8',
            'девять': '9',
            'тысяча': ''}
    
    def normalize(self, text):
        words = text.lower().split()
        digits = re.findall(r'\d', text)
        if digits:
            return ''.join(digits)

    
    def denormalize(self, digits):
        return ' '.join(str(d) for d in digits)

class DigitDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_length=148, augmenter=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length
        self.normalizer = RussianNumberNormalizer()
        self.augmenter = augmenter
        self.data['speaker_id'] = self.data.iloc[:, 0].str.extract(r'(spk\d+)')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = f"{self.root_dir}/{self.data.iloc[idx, 0]}"
        raw_transcription = str(self.data.iloc[idx, 1])
        speaker_id = self.data.iloc[idx]['speaker_id']
        normalized_transcription = self.normalizer.normalize(raw_transcription)
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if self.augmenter is not None:
            waveform = self.augmenter(waveform)
        
        spectrogram = self.transform(waveform)
        
        target = [int(d) for d in normalized_transcription]
        
        return spectrogram, target, speaker_id

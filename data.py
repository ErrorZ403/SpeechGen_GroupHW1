import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torchaudio.transforms import MelSpectrogram, Resample

def collate_fn(batch):
    max_len = min(148, max(spec.shape[1] for spec, _ in batch))
    
    spectrograms = []
    for spec, _ in batch:
        if spec.shape[1] > max_len:
            spec = spec[:, :max_len]
        spectrograms.append(spec.squeeze(0).T)
    
    targets = [torch.tensor(item[1]) for item in batch]
    spec_lengths = torch.tensor([min(spec.shape[0], max_len) for spec in spectrograms])
    target_lengths = torch.tensor([len(t) for t in targets])
    
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return spectrograms, targets, spec_lengths, target_lengths

class DigitDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_length=148):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = f"{self.root_dir}/{self.data.iloc[idx, 0]}"
        transcription = str(self.data.iloc[idx, 1])
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            resampler = Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        spectrogram = self.transform(waveform)
        
        if spectrogram.shape[2] > self.max_length:
            spectrogram = spectrogram[:, :, :self.max_length]
        
        target = [int(d) for d in transcription]
        
        return spectrogram, target

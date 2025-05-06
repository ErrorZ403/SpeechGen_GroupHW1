import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model import DigitHybridModel
from data import RussianNumberNormalizer
from torchaudio.transforms import MelSpectrogram, Resample


class AudioProcessor:
    def __init__(self, sample_rate=16000, n_mels=60, n_fft=400, hop_length=160):
        self.sample_rate = sample_rate
        self.transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
    def process(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != self.sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        spectrogram = self.transform(waveform)
        
        if spectrogram.shape[2] > 148:
            spectrogram = spectrogram[:, :, :148]
            
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)
        
        return spectrogram


class DigitRecognitionInference:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.processor = AudioProcessor()
        
        self.normalizer = RussianNumberNormalizer()
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        hparams = checkpoint['hyper_parameters']
        
        model = DigitHybridModel(
            input_dim=hparams.get('input_dim', 60),  # n_mels
            hidden_dim=hparams.get('hidden_dim', 512),
            output_dim=hparams.get('output_dim', 11),
            model_type=hparams.get('model_type', 'hybrid'),
            rnn_type=hparams.get('rnn_type', 'lstm')
        ).to(self.device)
        
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
        
    def predict_single(self, audio_path):
        spectrogram = self.processor.process(audio_path)
        spectrogram = spectrogram.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(spectrogram)
            outputs = outputs.log_softmax(2)
            
        spec_length = torch.tensor([spectrogram.size(1)])
        adjusted_length = (spec_length / 4).long()
        decoded_sequence = self.model.greedy_decode(outputs, adjusted_length)[0]
        
        return decoded_sequence
    
    def process_test_dataset(self, test_dir, output_csv):
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        
        results = []
        for filename in tqdm(test_files, desc="Processing test files"):
            file_path = os.path.join(test_dir, filename)
            prediction = self.predict_single(file_path)
            
            results.append({
                'file_name': filename,
                'predicted_digit_sequence': prediction
            })
        
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print(f"Saved predictions to {output_csv}")


def main():
    model_path = "checkpoints/model-best.ckpt"
    test_dir = "data/test"
    output_csv = "submission.csv"
    
    inference = DigitRecognitionInference(model_path)
    
    inference.process_test_dataset(test_dir, output_csv)


if __name__ == "__main__":
    main()
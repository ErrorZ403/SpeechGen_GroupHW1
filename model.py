import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from ptflops import get_model_complexity_info
from jiwer import wer


class DigitHybridModel(pl.LightningModule):
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=11, model_type='hybrid', rnn_type='lstm'):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.rnn_type = rnn_type
        
        if model_type in ['cnn', 'hybrid']:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=11, stride=2, padding=5),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(128, 128, kernel_size=11, stride=2, padding=5),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            cnn_output_dim = 128
        else:
            self.conv = None
            cnn_output_dim = input_dim
            
        if model_type in ['rnn', 'hybrid']:
            if rnn_type == 'lstm':
                self.rnn = nn.LSTM(cnn_output_dim, hidden_dim, num_layers=2, 
                                 bidirectional=True, batch_first=True, dropout=0.2)
            else:
                self.rnn = nn.GRU(cnn_output_dim, hidden_dim, num_layers=2,
                                bidirectional=True, batch_first=True, dropout=0.2)
            rnn_output_dim = hidden_dim * 2
        else:
            self.rnn = None
            rnn_output_dim = cnn_output_dim
            
        self.fc = nn.Linear(rnn_output_dim, output_dim)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        if x.size(1) > 148:
            x = x[:, :148, :]
            
        if self.model_type in ['cnn', 'hybrid']:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)

        if self.model_type in ['rnn', 'hybrid']:
            x, _ = self.rnn(x)
            
        x = self.fc(x)
       
        return x

    def greedy_decode(self, log_probs, input_lengths):
        _, decoded = log_probs.max(2)
        decoded = decoded.cpu().numpy()
        
        decoded_sequences = []
        for i in range(len(decoded)):
            seq = decoded[i][:input_lengths[i]]
            
            prev_token = None
            decoded_seq = []
            for token in seq:
                if token != 0 and token != prev_token: 
                    decoded_seq.append(str(token))
                prev_token = token
            
            decoded_sequences.append(''.join(decoded_seq))
            
        return decoded_sequences

    def training_step(self, batch, batch_idx):
        spectrograms, targets, spec_lengths, target_lengths = batch
        outputs = self(spectrograms)
        outputs = outputs.log_softmax(2)
        
        adjusted_lengths = (spec_lengths / 4).long()
        loss = self.criterion(outputs.permute(1, 0, 2), targets, adjusted_lengths, target_lengths)
        
        if batch_idx % 100 == 0:
            decoded_sequences = self.greedy_decode(outputs, adjusted_lengths)
            targets_np = targets.cpu().numpy()
            for i in range(min(3, len(decoded_sequences))):  # Log first 3 examples
                true = ''.join([str(t) for t in targets_np[i][:target_lengths[i]]])
                pred = decoded_sequences[i]
                self.logger.experiment.add_text(f'train/prediction_{i}', 
                                              f'True: {true}\nPred: {pred}', 
                                              self.global_step)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if batch_idx % 100 == 0:
            self.logger.experiment.add_image('train/spectrogram', 
                                           spectrograms[0].unsqueeze(0), 
                                           self.global_step)
        
        return loss

    def validation_step(self, batch, batch_idx):
        spectrograms, targets, spec_lengths, target_lengths = batch
        outputs = self(spectrograms)
        outputs = outputs.log_softmax(2)
        
        adjusted_lengths = (spec_lengths / 4).long()
        loss = self.criterion(outputs.permute(1, 0, 2), targets, adjusted_lengths, target_lengths)
        
        decoded_sequences = self.greedy_decode(outputs, adjusted_lengths)
        targets = targets.cpu().numpy()
        
        wers = []
        correct_predictions = 0
        total_predictions = len(decoded_sequences)
        
        for i in range(len(decoded_sequences)):
            true = ''.join([str(t) for t in targets[i][:target_lengths[i]]])
            pred = decoded_sequences[i]
            wers.append(wer(true, pred))
            
            if true == pred:
                correct_predictions += 1
            if i < 3:
                self.logger.experiment.add_text(f'val/prediction_{i}', 
                                              f'True: {true}\nPred: {pred}\nWER: {wers[-1]:.4f}', 
                                              self.global_step)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_wer', sum(wers) / len(wers), on_epoch=True, prog_bar=True)
        self.log('val_accuracy', correct_predictions / total_predictions, on_epoch=True)
        
        if batch_idx % 10 == 0:
            self.logger.experiment.add_image('val/spectrogram', 
                                           spectrograms[0].unsqueeze(0), 
                                           self.global_step)
            
            self.logger.experiment.add_histogram('val/output_distribution', 
                                               outputs[0], 
                                               self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_wer"
            }
        }
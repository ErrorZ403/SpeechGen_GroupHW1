import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics
from jiwer import wer

from data import RussianNumberNormalizer

def sep_conv(inp, outp, k=5, s=1, p=2, dil=1):
    pad = (k // 2) * dil
    return nn.Sequential(
        nn.Conv1d(inp, inp, k, s, pad, groups=inp, dilation=dil, bias=False),
        nn.Conv1d(inp, outp, 1, bias=False),
        nn.BatchNorm1d(outp),
        nn.SiLU(),
        nn.Dropout(0.15),
    )

class DigitHybridModel(pl.LightningModule):
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=11, model_type='hybrid', rnn_type='lstm', conv_type='simple'):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.rnn_type = rnn_type
        
        if model_type in ['cnn', 'hybrid']:
            if conv_type == 'simple':
                self.conv = nn.Sequential(
                    nn.Conv1d(input_dim, 128, kernel_size=11, stride=2, padding=5),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Conv1d(128, 128, kernel_size=11, stride=2, padding=5),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                self.compute_len = self._cnn_out_len
            else:
                self.conv = nn.Sequential(
                    sep_conv(input_dim, 32, 5, 2, 2),  
                    sep_conv(32, 64, 5, 2, 2), 
                    sep_conv(64, 128, dil=2),
                    sep_conv(128, 128, dil=2),
                )
                self.compute_len = self._len_after_cnn
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
        self.criterion = nn.CTCLoss(blank=10, zero_infinity=True)

    def _cnn_out_len(self, lens):
        for _ in range(2):                      
            lens = (lens + 2*5 - (11-1) - 1) // 2 + 1
        return lens
    
    def _len_after_cnn(self, l):
        for s in (2,2):
            l = (l+1)//s
        return l

    def forward(self, x):            
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
                if token != 10 and token != prev_token: 
                    decoded_seq.append(str(token))
                prev_token = token
            
            decoded_sequences.append(''.join(decoded_seq))
            
        return decoded_sequences

    def training_step(self, batch, batch_idx):
        spectrograms, targets, spec_lengths, target_lengths, speaker_ids = batch
        outputs = self(spectrograms)
        outputs = outputs.log_softmax(2)
        
        adjusted_lengths = self.compute_len(spec_lengths.clone())
        loss = self.criterion(outputs.permute(1, 0, 2), targets, adjusted_lengths, target_lengths)

        log_probs = F.log_softmax(outputs, dim=2)
        assert (targets[torch.arange(len(targets)), target_lengths-1] != 11).all()
        assert (targets[torch.arange(len(targets)).unsqueeze(1),
                torch.arange(targets.shape[1]).unsqueeze(0)]
        .masked_select(torch.arange(targets.shape[1]) >= target_lengths.unsqueeze(1))
        == 11).all(), "padding zone has non‑PAD values"
        
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
        spectrograms, targets, spec_lengths, target_lengths, speaker_ids = batch
        outputs = self(spectrograms)
        outputs = outputs.log_softmax(2)
        
        adjusted_lengths = self.compute_len(spec_lengths.clone())
        loss = self.criterion(outputs.permute(1, 0, 2), targets, adjusted_lengths, target_lengths)
        
        decoded_sequences = self.greedy_decode(outputs, adjusted_lengths)
        targets = targets.cpu().numpy()
        normalizer = RussianNumberNormalizer()
        
        wers = []
        correct_predictions = 0
        total_predictions = len(decoded_sequences)
        cers = []

        for i, (pred, true_seq, spk_id) in enumerate(zip(decoded_sequences, targets, speaker_ids)):
            true = ''.join([str(t) for t in true_seq[:target_lengths[i]]])
            wers.append(wer(true, pred))
            cer = self.compute_cer(true, pred)
            cers.append(cer)
            
            self.log(f'val_cer_speaker_{spk_id}', cer, on_epoch=True)
            
            if i < 3:
                denorm_true = normalizer.denormalize(true)
                denorm_pred = normalizer.denormalize(pred)
                
                self.logger.experiment.add_text(
                    f'val/prediction_{i}', 
                    f'Speaker: {spk_id}\n'
                    f'True (normalized): {true}\n'
                    f'Pred (normalized): {pred}\n'
                    f'True (words): {denorm_true}\n'
                    f'Pred (words): {denorm_pred}\n'
                    f'CER: {cer:.4f}', 
                    self.global_step
                )
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_cer', sum(cers) / len(cers), on_epoch=True, prog_bar=True)
        self.log('val_wer', sum(wers) / len(wers), on_epoch=True, prog_bar=True)

        if batch_idx % 10 == 0:
            self.logger.experiment.add_image('val/spectrogram', 
                                           spectrograms[0].unsqueeze(0), 
                                           self.global_step)
            
            self.logger.experiment.add_histogram('val/output_distribution', 
                                               outputs[0], 
                                               self.global_step)

    def compute_cer(self, reference, hypothesis):
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        d = [[0 for _ in range(len(hyp_chars) + 1)] for _ in range(len(ref_chars) + 1)]
        
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
                
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

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
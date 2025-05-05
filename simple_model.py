import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl



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

    def _cnn_out_len(self, lens):
        for _ in range(2):                       # two Conv1d layers
            lens = (lens + 2*5 - (11-1) - 1) // 2 + 1
        return lens

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
                if token != 0 and token != prev_token: 
                    decoded_seq.append(str(token))
                prev_token = token
            
            decoded_sequences.append(''.join(decoded_seq))
            
        return decoded_sequences
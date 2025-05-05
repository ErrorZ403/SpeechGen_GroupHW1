import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import DigitHybridModel
from data import DigitDataset, collate_fn, AudioAugmenter
from torchaudio.transforms import MelSpectrogram


def main():
    pl.seed_everything(42)
    
    n_mels = 40
    sample_rate = 16000
    n_fft = 400
    hop_length = 160

    hidden_dim = 384
    output_dim = 11
    model_type = 'hybrid'
    rnn_type = 'lstm'
    
    transform = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    augmenter = AudioAugmenter(sample_rate=sample_rate)
    
    train_dataset = DigitDataset(csv_file='data/train/train.csv', root_dir='data/train/', transform=transform, augmenter=None)
    val_dataset = DigitDataset(csv_file='data/dev/dev.csv', root_dir='data/dev/', transform=transform, augmenter=None)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = DigitHybridModel(
        input_dim=n_mels,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_type=model_type,
        rnn_type=rnn_type
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_cer',
        mode='min',
        save_top_k=3,
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_cer:.4f}'
    )
    
    logger = TensorBoardLogger('logs', name='digit_recognition')
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main()

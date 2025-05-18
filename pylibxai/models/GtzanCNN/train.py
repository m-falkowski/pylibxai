import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision
import torchaudio
from tqdm.notebook import trange
from sklearn.metrics import accuracy_score, confusion_matrix

from pylibxai.models.GtzanCNN.model import CNN
from pylibxai.models.GtzanCNN.preprocessing import TRANSFORM
from pylibxai.utils import get_install_path

GTZAN_ROOT_DIR = get_install_path() / "pylibxai" / "datasets" / "GTZAN" / "Data"
SEED = 123
EPOCHS = 700
BATCH_SIZE = 128
LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)


gtzan_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
labels = {i: label for i, label in enumerate(gtzan_genres)}
label_to_id = {label: i for i, label in enumerate(gtzan_genres)}


class GtzanDataset(Dataset):
    def __init__(self, metadata_file, gtzan_root, transform=None, sr=22050):
        self.metadata = pd.read_csv(metadata_file)
        self.audio_dir = os.path.join(gtzan_root, "genres_original")
        self.transform = transform
        self.sr = sr

    def __len__(self):
        return len(self.metadata)

    def pad_or_truncate_waveform(self, wav, target_len):
        current_len = wav.shape[-1]
        if current_len < target_len:
            pad_amt = target_len - current_len
            wav = F.pad(wav, (0, pad_amt))  # pad end with zeros
        elif current_len > target_len:
            wav = wav[:, :target_len]  # truncate
        return wav

    def __getitem__(self, idx):
        label = self.metadata.iloc[idx, -1]
        wav_path = os.path.join(self.audio_dir, label, self.metadata.iloc[idx, 0])
        label = label_to_id[self.metadata.iloc[idx, -1]]

        wav, _ = torchaudio.load(wav_path, normalize=True)
        wav = self.pad_or_truncate_waveform(wav, target_len=self.sr * 30)
            
        if self.transform:
            data = self.transform(wav)
        else:
            data = torch.from_numpy(wav).float()

        return data, label

def main():
    print('main start')
    print(f'{torch.cuda.is_available()=}')
    
    metadata = f"{GTZAN_ROOT_DIR}/features_30_sec_mod.csv"
    dataset = GtzanDataset(metadata, GTZAN_ROOT_DIR, transform=TRANSFORM)

    val_pc = 0.2

    val_size = int(len(dataset)*val_pc)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 64

    train_loader = DataLoader(train_ds, batch_size, shuffle = True)
    valid_loader = DataLoader(val_ds, batch_size, shuffle = False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cnn = CNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
    valid_losses = []
    EPOCHS = 30
    progress = trange(EPOCHS, desc="Training")

    def train():
        for epoch in progress:
            losses = []

            # Train
            cnn.train()
            for (wav, genre_index) in train_loader:
                wav = wav.to(device)
                genre_index = genre_index.to(device)

                # Forward
                out = cnn(wav)
                loss = loss_function(out, genre_index)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print('Epoch: [%d/%d], Train loss: %.4f' % (epoch+1, EPOCHS, np.mean(losses)))

            # Validation
            cnn.eval()
            y_true = []
            y_pred = []
            losses = []
            for wav, genre_index in valid_loader:
                wav = wav.to(device)
                genre_index = genre_index.to(device)

                # reshape and aggregate chunk-level predictions
                #print(f'{wav.shape}')
                #b, c, t = wav.size()
                #b, c, mel_bins, t = wav.size()
                logits = cnn(wav)
                #logits = logits.view(b, c, -1).mean(dim=1)
                loss = loss_function(logits, genre_index)
                losses.append(loss.item())
                _, pred = torch.max(logits.data, 1)

                # append labels and predictions
                y_true.extend(genre_index.tolist())
                y_pred.extend(pred.tolist())
            accuracy = accuracy_score(y_true, y_pred)
            valid_loss = np.mean(losses)
            print('Epoch: [%d/%d], Valid loss: %.4f, Valid accuracy: %.4f' % (epoch+1, EPOCHS, valid_loss, accuracy))

            # Save model
            valid_losses.append(valid_loss.item())
            if np.argmin(valid_losses) == epoch:
                print('Saving the best model at %d epochs!' % epoch)
                torch.save(cnn.state_dict(), 'gtzan_cnn.ckpt')

    if not os.path.exists('gtzan_cnn.ckpt'):
        print('Saving model into gtzan_cnn.ckpt')
        train()
        torch.save(cnn.state_dict(), 'gtzan_cnn.ckpt')
    else:
        print('Model is already trained, skipping')
        S = torch.load('gtzan_cnn.ckpt')
        cnn.load_state_dict(S)

if __name__ == "__main__":
    main()
    
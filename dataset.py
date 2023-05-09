import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels=None, transform=None):
        super().__init__()
        
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, index):
        spectrogram = self.spectrograms[index]

        if self.transform:
            spectrogram = self.transform(spectrogram)
        else:
            spectrogram = torch.from_numpy(spectrogram).float()
        
        # If there are no labels, only spectrograms are returned
        if self.labels is None:
            return spectrogram
        
        label = self.labels[index]
        label = torch.from_numpy(np.array(label)).long()

        return spectrogram, label
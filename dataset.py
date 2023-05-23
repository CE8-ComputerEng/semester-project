import numpy as np
import torch
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels=None, transform=None):
        super().__init__()
        
        self.spectrograms = spectrograms
        # labels contain a list for each sample, and this list contains the labels, filename, and time info.
        # labels = [[[label1, label2, label3], filename, time_info], [[label1, label2], filename, time_info], ...]
        # We now want to extract the labels from this list of lists.
        # End result sould be a numpy array of shape (num_samples, num_labels)

        self.labels = labels
        #print(self.labels.shape)
        #self.labels = labels[:, 0]
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
        
        label = self.labels[index].label
        label = torch.from_numpy(np.array(label)).long()
        start = self.labels[index].start
        end = self.labels[index].end
        filename = self.labels[index].filename

        return spectrogram, label, start, end, filename
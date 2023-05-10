
# ## Imports


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from sklearn.model_selection import train_test_split
from torchsummary import summary
import glob
import tqdm
from dataset import SpectrogramDataset
import utils
import dataimporter
from utils import plot_confusion_matrix


# ## CNN model


class CNNClassifier(pl.LightningModule):
    def __init__(self, classes):
        super(CNNClassifier, self).__init__()
        
        self.classes = classes
        num_classes = len(classes)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(128 * 5 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = nn.Threshold(0.5, 0)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        
        self.test_confusion_matrix = np.zeros((num_classes, num_classes))
        
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        #print(x.shape)
        x = x.view(-1, 128 * x.shape[2] * x.shape[3])
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        
        # Prediction should also be able to handle multi label
        _, predicted = self.threshold(y_hat)
        print(predicted)
        #_, predicted = torch.max(y_hat, 1)
        
        # multi label accuracy
        accuracy = (predicted == y).sum().item() / (len(y) * len(self.classes))
        print(accuracy)
        #accuracy = (predicted == y).sum().item() / len(y)
        self.log('train_acc', accuracy, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.type(torch.FloatTensor).to(self.device)

        #print(y_hat.shape)
        #print(y.shape)
        """average_loss = 0
        for sample, label in zip(y_hat, y):
            print(sample)
            print(label)
            loss = self.criterion(sample, label)
            print(loss)
            average_loss += loss
        loss = average_loss / len(y_hat)"""
        loss = self.criterion(y_hat, y)
        print(loss)
        self.log('val_loss', loss)
        
        # Take the 
        print(predicted)
        
        #_, predicted = torch.max(y_hat, 1)
        #accuracy = (predicted == y).sum().item() / len(y)
        self.log('val_acc', accuracy)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        
        _, predicted = torch.max(y_hat, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        self.log('test_acc', accuracy)
        
        confusion_matrix = self.confusion_matrix(predicted, y)
        self.test_confusion_matrix += confusion_matrix.cpu().numpy()
        
    def on_test_epoch_end(self):
        plot_confusion_matrix(self.test_confusion_matrix, self.classes, filename='confusion_matrix.png')
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        y_hat = self(x)
        
        _, predicted = torch.max(y_hat, 1)
        return predicted

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

def main():
    # ## Import data and labels
    # 


    CLASSES = ['BACKGROUND','JUMP','BOAT','SEAGUL_SCREAM','BRIDGE','AMBULANCE','SCOOTER','PEE','OBJECT_SPLASH','UFO','IDLE_MOTOR','SEAGUL_SPLASH','VOICE', 'SWIM']
    NPDATAPATH = 'data/measurment-2/data.npy'
    NPLABELPATH = 'data/measurment-2/labels.npy'
    DATAPATH = 'data/measurment-2/clips/wav/'
    LABELPATH = 'data/measurment-2/clips/txt/'
    data_np, labels_np, data_size = dataimporter.import_data(NPDATAPATH, NPLABELPATH, DATAPATH, LABELPATH, CLASSES)



    # ## Just for fun: spectrogram plot loop
    # Uncomment if you want to see the plot of the spectrograms in a loop.


    utils.plot_audio_spectogram(data_np[99])
    print(labels_np[99])
    #%matplotlib qt
    #utils.loop_plot_audio_spectogram(data_np)
    #%matplotlib inline


    # ## Prepare Training, validation and test data


    TRAINING_RATIO = 0.8
    VALIDATION_RATIO = 0.1
    TEST_RATIO = 0.1

    if TRAINING_RATIO + VALIDATION_RATIO + TEST_RATIO != 1:
        raise ValueError('Training, validation, and test ratios must sum to 1.')

    train_size = int(TRAINING_RATIO * len(data_np))
    val_size = int(VALIDATION_RATIO * len(data_np))
    test_size = len(data_np) - train_size - val_size

    train_spectrograms, val_spectrograms, train_labels, val_labels = train_test_split(data_np, labels_np, test_size=val_size, random_state=42)
    train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(train_spectrograms, train_labels, test_size=test_size, random_state=42)

    print('Training samples:', train_spectrograms.shape[0])
    print('Validation samples:', val_spectrograms.shape[0])
    print('Test samples:', test_spectrograms.shape[0])


    # ## Define transforms


    # Define data transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transforms here
    ])


    # ## Make datasets


    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Create datasets and dataloaders
    train_dataset = SpectrogramDataset(train_spectrograms, train_labels, transform=transform)
    val_dataset = SpectrogramDataset(val_spectrograms, val_labels, transform=transforms.ToTensor())
    test_dataset = SpectrogramDataset(test_spectrograms, test_labels, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    # ## Define classes, and summary of the model

    model = CNNClassifier(classes=CLASSES)
    print("Data size: ", data_size)

    summary(model, (1 , data_size[0], data_size[1]))


    # ## Define trainer


    MAX_EPOCHS = 30
    VERSION = 'cnn_v2_epoch-30-v2'

    accelerator = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
    elif torch.backends.mps.is_available():
        accelerator = 'cpu'  # MPS is not implemented in PyTorch yet

    tb_logger = loggers.TensorBoardLogger('.', version=VERSION)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, save_last=True, filename='best-{epoch}-{val_acc:.2f}')

    trainer = Trainer(accelerator=accelerator, max_epochs=MAX_EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback])


    # ## Start training


    trainer.fit(model, train_loader, val_loader)


    # Load previously trained model
    CHECKPOINT_PATH = f'lightning_logs/{VERSION}/checkpoints/best-epoch=23-val_acc=0.95.ckpt'

    model = CNNClassifier.load_from_checkpoint(CHECKPOINT_PATH, classes=CLASSES)
    print(f'Model size: {os.path.getsize(CHECKPOINT_PATH) / 1e6} MB')


    trainer.test(model, test_loader) # This not the challenge, test set


    spectrograms_to_predict = np.load('data/test.npy')

    print('Test spectrogram to predict shape:', spectrograms_to_predict.shape)
    print('Test spectrogram to predict dtype:', spectrograms_to_predict.dtype)


    test_to_predict_dataset = SpectrogramDataset(spectrograms_to_predict, transform=transforms.ToTensor())
    test_to_predict_loader = DataLoader(test_to_predict_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    predictions = trainer.predict(model, test_to_predict_loader)
    predictions = np.concatenate(predictions).astype(int)


    np.savetxt('predictions.txt', predictions, delimiter='\n', fmt='%d')


if __name__ == '__main__':
    main()
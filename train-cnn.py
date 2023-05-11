
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
from utils import plot_confusion_matrix, plot_multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
#torch.autograd.set_detect_anomaly(True)
# ## CNN model

#   TODO:
# - Try with smaller audio clips


class CNNClassifier(pl.LightningModule):
    def __init__(self, classes):
        super(CNNClassifier, self).__init__()
        
        self.classes = classes
        num_classes = len(classes)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(256 * 32 * 1, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        #self.threshold = nn.Threshold(0.5, 0)
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss()
        #self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.confusion_matrix = None
        self.test_confusion_matrix = None
        #self.test_confusion_matrix = np.zeros((num_classes, num_classes))
        
    def single_label(self, y_new):
        # Make y_new only contain 0 or 1, where 0 is Background and 1 is the other classes
        y_single = torch.zeros((len(y_new), 1)).to(self.device)
        for i in range(len(y_new)):
            for j in range(len(y_new[i])):
                if y_new[i][j] != 0:
                    y_single[i] = 1
                else:
                    y_single[i] = 0
        return y_single
                    
    def multi_label_threshold(self, y_hat):
        # Take y_hat and set all values below 0.5 to 0 and all values above 0.5 to 1
        # If none of the values are above 0.5, set the highest value to 1
        predicted = y_hat.clone().detach().type(torch.FloatTensor).to(self.device)
        highest = 0
        highest_index = 0
        found_one = False
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                #print(predicted[i][j])
                if predicted[i][j] < 0.3:
                    predicted[i][j] = 0
                    if predicted[i][j] > highest:
                        highest = predicted[i][j]
                        highest_index = j
                else:
                    predicted[i][j] = 1
                    found_one = True
            if not found_one:
                predicted[i][highest_index] = 1
                #print(predicted[i])
            highest = 0
            highest_index = 0
            found_one = False
            
        return predicted
        
    def default_step_work(self, batch, batch_idx, loss_log, acc_log):
        x, y = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        loss = self.criterion(y_hat, y_new)
        #print(loss)
        self.log(loss_log, loss)
        predicted = self.multi_label_threshold(y_hat)
        #_, predicted = torch.max(y_hat, 1)
        #print(predicted)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log(acc_log, accuracy)
        return loss
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        #print(x.shape)
        x = x.view(-1, 256 * x.shape[2] * x.shape[3])
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.default_step_work(batch, batch_idx, 'train_loss', 'train_acc')
        """x, y = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        loss = self.criterion(y_hat, y_new)
        #print(loss)
        self.log('train_loss', loss)


        #_, predicted = torch.max(y_hat, 1)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log('train_acc', accuracy)"""
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.default_step_work(batch, batch_idx, 'val_loss', 'val_acc')
        """x, y = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        loss = self.criterion(y_hat, y_new)
        #print(loss)
        self.log('val_loss', loss)
        
        #_, predicted = torch.max(y_hat, 1)
        #print(predicted)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log('val_acc', accuracy)"""
    
    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        
        loss = self.criterion(y_hat, y_new)
        #print(loss)
        self.log('test_loss', loss)
        predicted = self.multi_label_threshold(y_hat)
        #_, predicted = torch.max(y_hat, 1)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log('test_acc', accuracy)
        
        self.confusion_matrix = multilabel_confusion_matrix(y_new.cpu().numpy(), predicted.cpu().numpy())
        if self.test_confusion_matrix is None:
            self.test_confusion_matrix = self.confusion_matrix
        else:
            self.test_confusion_matrix += self.confusion_matrix
        #self.test_confusion_matrix += self.confusion_matrix

        #confusion_matrix = self.confusion_matrix(predicted, y)
        #self.test_confusion_matrix += confusion_matrix.cpu().numpy()

        
    def on_test_epoch_end(self):
        #plot_confusion_matrix(self.test_confusion_matrix, self.classes, filename='confusion_matrix.png')
        plot_multilabel_confusion_matrix(self.test_confusion_matrix, self.classes, filename='multilabel_confusion_matrix.png')
        #print(self.test_confusion_matrix)
        pass 
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


    CLASSES = ['BACKGROUND','JUMP','BOAT','SEAGUL_SCREAM','BRIDGE','AMBULANCE','SCOOTER','PEE','OBJECT_SPLASH','UFO','IDLE_MOTOR','SEAGUL_SPLASH','VOICE', 'SWIM', 'HORN', 'ROCK']
    NPDATAPATH = 'data/measurment-2/data.npy'
    NPLABELPATH = 'data/measurment-2/labels.npy'
    DATAPATH = 'data/measurment-2/clips/wav/'
    LABELPATH = 'data/measurment-2/clips/txt/'
    data_np, labels_np, data_size = dataimporter.import_data(NPDATAPATH, NPLABELPATH, DATAPATH, LABELPATH, CLASSES)

    # Plot the distribution of the labels

    class DistributionDataset:
        def __init__(self, data_np, labels_np, CLASSES):
            label_count = np.zeros(len(CLASSES))
            for data_np, labels_np in zip(data_np, labels_np):
                label_count += labels_np
            self.index = CLASSES
            self.values = label_count
        
    distribution = DistributionDataset(data_np, labels_np, CLASSES)
    #utils.plot_label_distribution(distribution)

    # Every sample that contains a label that is not JUMP, Should have BACKGROUND set to 0.
    for i in range(len(labels_np)):
        if labels_np[i].any() and labels_np[i][1] == 0:
            labels_np[i][0] = 0
        elif labels_np[i][1] == 1:
            labels_np[i][0] = 0
            
    distribution = DistributionDataset(data_np, labels_np, CLASSES)
    #utils.plot_label_distribution(distribution)

    # Randomly choose 500 IDLE_MOTOR and BOAT samples to keep, and discard the rest.
    
    
    
    # Now we limit the number of labels to 2, Background and Jump
    # If a samples contains both Background and Jump, it will be labeled as Jump.
    # Otherwise it will be labeled as Background.
    # All other labels will be removed.
    
    # Remove all labels except Background and Jump
    #labels_np = np.delete(labels_np, [2,3,4,5,6,7,8,9,10,11,12,13,14,15], axis=1)
    """for i in range(len(labels_np)):
        if labels_np[i][0] == 1 and labels_np[i][1] == 1:
            labels_np[i][0] = 0
            labels_np[i][1] = 1
            
    CLASSES = ['BACKGROUND', 'JUMP']
    distribution = DistributionDataset(data_np, labels_np, CLASSES)
    utils.plot_label_distribution(distribution)
    print("Number of JUMP samples: ", distribution.values[1])"""
    # ## Just for fun: spectrogram plot loop
    # Uncomment if you want to see the plot of the spectrograms in a loop.


    #utils.plot_audio_spectogram(data_np[99])
    #print(labels_np[99])
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

    MAX_EPOCHS = 40
    VERSION = 'huge_cnn_v2_epoch-40-multi_label_small_clip'

    accelerator = None
    if torch.cuda.is_available():
        accelerator = 'gpu'
    elif torch.backends.mps.is_available():
        accelerator = 'cpu'  # MPS is not implemented in PyTorch yet

    tb_logger = loggers.TensorBoardLogger('.', version=VERSION)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, save_last=True, filename='best-{epoch}-{val_acc:.2f}')

    trainer = Trainer(accelerator=accelerator, max_epochs=MAX_EPOCHS, logger=tb_logger, callbacks=[checkpoint_callback])


    # ## Start training

    #trainer.fit(model, train_loader, val_loader)

    # Load previously trained model
    CHECKPOINT_PATH = f'lightning_logs/{VERSION}/checkpoints/best-epoch=38-val_acc=0.81.ckpt'

    model = CNNClassifier.load_from_checkpoint(CHECKPOINT_PATH, classes=CLASSES)
    print(f'Model size: {os.path.getsize(CHECKPOINT_PATH) / 1e6} MB')

    trainer.test(model, test_loader) # This not the challenge, test set
    #print('Test set accuracy:', model.log_dict['test_acc'])
    #print('Test set loss   :', model.log_dict['test_loss'])

if __name__ == '__main__':
    main()
    
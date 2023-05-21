
# ## Imports
# tensorboard --logdir lightning_logs

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
import cnn_dataimporter
import matplotlib.pyplot as plt
import librosa
from data_splitter import split_data_val_train
#torch.autograd.set_detect_anomaly(True)
# ## CNN model
class DistributionDataset:
    def __init__(self, data_np, labels_np, CLASSES):
        label_count = np.zeros(len(CLASSES))
        for data_np, labels_np in zip(data_np, labels_np):
            label_count += labels_np.label
        self.index = CLASSES
        self.values = label_count
                
def main():
    # ## Import data and labels
    # 
    # Possible spectrogram types are "stft", "mel", "mfcc", and "pncc"
    spectrogram_type = "mel"
    CLASSES = ['JUMP','BOAT','SEAGUL_SCREAM','BRIDGE','SCOOTER','PEE','OBJECT_SPLASH','UFO','IDLE_MOTOR','SEAGUL_SPLASH','VOICE', 'SWIM']
    FULL_TRAIN_DATASET_PATH = 'data/full_data/'+spectrogram_type+'/train_data.npy'
    FULL_TRAIN_LABEL_PATH = 'data/full_data/'+spectrogram_type+'/train_labels.npy'
    FULL_VAL_DATASET_PATH = 'data/full_data/'+spectrogram_type+'/val_data.npy'
    FULL_VAL_LABEL_PATH = 'data/full_data/'+spectrogram_type+'/val_labels.npy'
    
    # Check if the full dataset has already been created
    if not os.path.exists(FULL_TRAIN_DATASET_PATH) or not os.path.exists(FULL_TRAIN_LABEL_PATH) or not os.path.exists(FULL_VAL_DATASET_PATH) or not os.path.exists(FULL_VAL_LABEL_PATH) :
        # Load the data and labels
    
        TRAIN_NPDATAPATH = 'data/training/'+spectrogram_type+'/data.npy'
        TRAIN_NPLABELPATH = 'data/training/'+spectrogram_type+'/labels.npy'
        TRAIN_DATAPATH = 'data/training/clips/wav/'
        TRAIN_LABELPATH = 'data/training/clips/txt/'
        TRAIN_TIMEPATH = 'data/training/clips/time/'
        train_data_np, train_labels_np, train_data_size = cnn_dataimporter.import_data(TRAIN_NPDATAPATH, TRAIN_NPLABELPATH, TRAIN_DATAPATH, TRAIN_LABELPATH, TRAIN_TIMEPATH, CLASSES, spectrogram_type)

        TEST_NPDATAPATH = 'data/test/'+spectrogram_type+'/data_jump.npy'
        TEST_NPLABELPATH = 'data/test/'+spectrogram_type+'/labels_jump.npy'
        TEST_DATAPATH = 'data/test/clips/wav/'
        TEST_LABELPATH = 'data/test/clips/txt/'
        TEST_TIMEPATH = 'data/test/clips/time/'
        # Plot the distribution of the labels
        test_data_np, test_labels_np, test_data_size = cnn_dataimporter.import_data(TEST_NPDATAPATH, TEST_NPLABELPATH, TEST_DATAPATH, TEST_LABELPATH,TEST_TIMEPATH, CLASSES, spectrogram_type)
        
        # Concatenate the test data and train data
        train_data_np = np.concatenate((train_data_np, test_data_np), axis=0)
        train_labels_np = np.concatenate((train_labels_np, test_labels_np), axis=0)
        print("train_data_np.shape = ", train_data_np.shape)
        print("train_labels_np.shape = ", train_labels_np.shape)
    
        c = 0
        #for single_class, i in zip(CLASSES, range(len(CLASSES))):
        for label in test_labels_np:
            # check if a clip only contains IDLE_MOTOR
            if label.label[10] == 1:
                if label.label[2]:
                    if label.label.sum() == 2:
                        c += 1
        print("Number of IDLE_MOTOR + BAOT clips = ", c)
        c = 0
            
        distribution = DistributionDataset(train_data_np, train_labels_np, CLASSES)
        print("Training data value distribution = ", distribution.values)
        #jump_distribution = DistributionDataset(test_data_np, test_labels_np, CLASSES)


        # ## Prepare Training, validation and test data
        TRAINING_RATIO = 0.8
        VALIDATION_RATIO = 0.1
        TEST_RATIO = 0.1

        if TRAINING_RATIO + VALIDATION_RATIO + TEST_RATIO != 1:
            raise ValueError('Training, validation, and test ratios must sum to 1.')

        """train_size = int(TRAINING_RATIO * len(train_data_np))
        val_size = int(VALIDATION_RATIO * len(train_data_np))
        test_size = len(train_data_np) - train_size - val_size

        train_spectrograms, val_spectrograms, train_labels, val_labels = train_test_split(train_data_np, train_labels_np, test_size=val_size, random_state=42)
        #train_spectrograms, test_spectrograms, train_labels, test_labels = train_test_split(train_spectrograms, train_labels, test_size=test_size, random_state=42)

        print('Training samples:', train_spectrograms.shape[0])
        print('Validation samples:', val_spectrograms.shape[0])
        #print('Test samples:', test_spectrograms.shape[0])"""
        train_spectrograms, train_labels, val_spectrograms, val_labels = split_data_val_train(train_data_np, train_labels_np, CLASSES, TRAIN_MAX_SAMPLES_PER_CLASS=4000)
        # Save the full dataset and labels
        np.save(FULL_TRAIN_DATASET_PATH, train_spectrograms)
        np.save(FULL_TRAIN_LABEL_PATH, train_labels)
        np.save(FULL_VAL_DATASET_PATH, val_spectrograms)
        np.save(FULL_VAL_LABEL_PATH, val_labels)
    else:
        train_spectrograms = np.load(FULL_TRAIN_DATASET_PATH)
        train_labels = np.load(FULL_TRAIN_LABEL_PATH, allow_pickle=True)
        val_spectrograms = np.load(FULL_VAL_DATASET_PATH)
        val_labels = np.load(FULL_VAL_LABEL_PATH, allow_pickle=True)
        
    
    
    training_distribution = DistributionDataset(train_spectrograms, train_labels, CLASSES)
    print("Training data value distribution after split = ", training_distribution.values)
    #utils.plot_label_distribution(training_distribution, filename='train_distribution.png')
    
    validation_distribution = DistributionDataset(val_spectrograms, val_labels, CLASSES)
    print("Validation data value distribution = ", validation_distribution.values)
    #utils.plot_label_distribution(validation_distribution, filename='val_distribution.png')
    
    # plot 10 random spectrograms
    """plt.figure(figsize=(10, 10))
    for i in range(10):
        index = np.random.randint(0, len(train_spectrograms))
        plt.imshow(train_spectrograms[index])
        plt.axis('off')
        plt.show()
        plt.close()"""
    # ## Define transforms
    # Define data transforms for data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transforms here
    ])

    
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Balance the training data. This can be done by using the pyto4rch WeightedRandomSampler
    # http://pytorch.org/docs/data.html#torch.utils.data.sampler.WeightedRandomSampler
    class_sample_count = training_distribution.values
    weights = 1. / torch.Tensor(class_sample_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, BATCH_SIZE)
    
    
    # ## Make datasets

    # Create datasets and dataloaders
    train_dataset = SpectrogramDataset(train_spectrograms, train_labels, transform=transform)
    val_dataset = SpectrogramDataset(val_spectrograms, val_labels, transform=transforms.ToTensor())
    #test_dataset = SpectrogramDataset(test_data_np, test_labels_np, transform=transforms.ToTensor())
    #jump_dataset = SpectrogramDataset(test_data_np, test_labels_np, transform=transforms.ToTensor())
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #jump_loader = DataLoader(jump_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    

    # ## Define classes, and summary of the model
    #if spectrogram_type == 'stft':
    #    from cnn_models.cnn_model import CNNClassifier
    #elif spectrogram_type == 'mel':
    #    from cnn_models.cnn_model_mel import CNNClassifier
    #elif spectrogram_type == 'mfcc':
    #    from cnn_models.cnn_model_mfcc import CNNClassifier
    from cnn_models.cnn_model import CNNClassifier
    print(train_spectrograms[0].shape)
    train_data_size = train_spectrograms[0].shape
    model = CNNClassifier(classes=CLASSES, sample_shape=train_spectrograms[0].shape, spectrogram_type=spectrogram_type)
    print("Data size: ", train_data_size)

    sum = summary(model, (1 , train_data_size[0], train_data_size[1]))
    #utils.save_summary_to_latex(model, (1 , train_data_size[0], train_data_size[1]),'CNN/model_summary_'+spectrogram_type+'.tex')
    # Save the layer names, activations and output sizes to a table for the report
    

    # ## Define trainer

    MAX_EPOCHS = 40
    VERSION = 'huge_cnn_v2_epoch-'+str(MAX_EPOCHS)+'-multi_label__multi_ch_medium_clip_with_all_jumps_sigmoid'+'_'+spectrogram_type + '_custom_loss'

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

    
    trainer = Trainer(accelerator=accelerator)
    # Load previously trained model
    
    # Best MEL = best-epoch=16-val_acc=0.55.ckpt
    # Best MFCC = best-epoch=30-val_acc=0.51.ckpt
    # BEST STFT = best-epoch=36-val_acc=0.63.ckpt
    CHECKPOINT_PATH = f'lightning_logs/{VERSION}/checkpoints/best-epoch=36-val_acc=0.53.ckpt'

    model = CNNClassifier.load_from_checkpoint(CHECKPOINT_PATH, classes=CLASSES, sample_shape=train_spectrograms[0].shape, spectrogram_type=spectrogram_type)
    print(f'Model size: {os.path.getsize(CHECKPOINT_PATH) / 1e6} MB')

    stats = trainer.test(model, val_loader)
    # Save the stats in a txt file
    with open(f'CNN/stats_'+spectrogram_type+'.txt', 'w') as f:
        f.write(str(stats))
    f.close()
    # Plot the confusion matrix just for the JUMP class
    #predicted = trainer.test(model, jump_loader)
    

if __name__ == '__main__':
    main()
    
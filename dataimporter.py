import numpy as np
import glob
import tqdm
import torchaudio
import os
def import_data(np_datapath, np_labelpath, datapath, labelpath):
    """This function will first try to load the data from the numpy file. If the file doesn't exist, it will create it from the audio files.
    Args:
        np_datapath (String): path to the numpy file containing the spectrograms
        np_labelpath (String): path to the numpy file containing the labels
        datapath (String): path to the folder containing the audio files
        labelpath (String): path to the folder containing the txt labels

    Returns:
        (np.array, np.array, tuple (w,h)): Returns the spectrograms and labels as numpy arrays and the size of a spectrogram in the form (height, width)
    """
    NPDATAPATH = np_datapath
    NPLABELPATH = np_labelpath
    # Try to load the data from the numpy file
    # See if the file exists
    if not os.path.exists(NPDATAPATH) or not os.path.exists(NPLABELPATH):
        print('Numpy file does not exist. Creating it now...')
                # If the file doesn't exist, create it.
        DATAPATH = datapath
        LABELPATH = labelpath

        data = glob.glob(DATAPATH + '*.wav')
        labels = glob.glob(LABELPATH + '*.txt')

        # Convert each wav file to a spectrogram and save it in a numpy array
        #data_np = np.empty((len(data)))
        #labels_np = np.empty((len(labels)))
        data_np = []
        labels_np = []
        # Use tqdm to show progress bar
        pbar = tqdm.tqdm(total=len(data))

        # Only test on 10 files for now
        #data = data[:100]
        i = 0
        for file, label in zip(data, labels):
            audio_file, rate_of_sample = torchaudio.load(file)
            spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=rate_of_sample, n_mels=128, n_fft=2048)(audio_file)
            spectrogram = torchaudio.transforms.AmplitudeToDB()(np.abs(spectrogram))
            data_np.append(spectrogram)
            labels_np.append(0) # TODO: Change this to the label
            i += 1
            """if data_np.size == 0:
                data_np[i] = spectrogram
                labels_np[i] = label
            else:
                data_np = np.concatenate((data_np, spectrogram), axis=0)
                labels_np = np.concatenate((labels_np, label), axis=0)"""
            pbar.update(1)
        pbar.close()
        # Convert list of numpy arrays to a single numpy array
        data_np = np.stack(data_np, dtype=np.float32, axis=0)
        labels_np = np.stack(labels_np, dtype=np.float32, axis=0)
        data_size = (data_np.shape[2], data_np.shape[3])
        print('Data shape:', data_np.shape)
        print('Labels shape:', labels_np.shape)
        
        # Data is now a numpy array of shape (num_files, 1, 128, 87)
        # Data should be of shape (num_files, 128, 87)
        
        data_np = np.squeeze(data_np, axis=1)
        print('Data shape:', data_np.shape)
        
        # Save the numpy array
        np.save(NPDATAPATH, data_np)
        np.save(NPLABELPATH, labels_np)
        return data_np, labels_np, data_size
    else:
        print('Numpy file exists. Loading it now...')
        data_np = np.load(NPDATAPATH)
        #data_size = (data_np.shape[2], data_np.shape[3])
        data_size = data_np.shape[1:]
        print('Data shape:', data_np.shape[1:])
        labels_np = np.load(NPLABELPATH)
        
        labels_np = labels_np.astype(int)
        data_np = data_np.astype(float)
        return data_np, labels_np, data_size
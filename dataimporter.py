import numpy as np
import glob
import tqdm
import torchaudio
import os
import utils
import librosa
def import_data(np_datapath, np_labelpath, datapath, labelpath, classes, all_classes = True):
    """This function will first try to load the data from the numpy file. If the file doesn't exist, it will create it from the audio files.
        Set all_classes to False to only use first found class in the label file.
    Args:
        np_datapath (String): path to the numpy file containing the spectrograms (should end with "/")
        np_labelpath (String): path to the numpy file containing the labels (should end with "/")
        datapath (String): path to the folder containing the audio files (should end with "/")
        labelpath (String): path to the folder containing the txt labels (should end with "/")
        classes (list): list of classes to use
        all_classes (bool, optional): Whether to use all classes or just the first. Defaults to True.

    Returns:
        (np.array, np.array, tuple (w,h)): Returns the spectrograms and labels as numpy arrays and the size of a spectrogram in the form (height, width)
    """
    # Create a dictionary to map the labels to numbers
    label_dict = {}
    i = 0
    for singel_class in classes:
        label_dict[singel_class] = i
        i += 1
    print(label_dict)

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
        data_np = []
        labels_np = []
        # Use tqdm to show progress bar
        pbar = tqdm.tqdm(total=len(data))

        # Only test on 10 files for now
        #data = data[:100]
        i = 0
        for file, label in zip(data, labels):
            #audio_file, rate_of_sample = torchaudio.load(file)
            audio, sr = librosa.load(file, sr=None, mono=False)
            spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
            spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), top_db=80, ref=1)
            # Normalize the spectrogram
            spectrogram = librosa.util.normalize(spectrogram)
            # Just check if spectrogram is normalized
            if np.max(spectrogram) > 1 or np.min(spectrogram) < -1:
                print('Spectrogram not normalized')

            #spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=rate_of_sample, n_mels=128, n_fft=2048)(audio_file)
            #spectrogram = torchaudio.transforms.AmplitudeToDB()(np.abs(spectrogram))
            data_np.append(spectrogram)
            # Go through each line in the label file and check if it contains a BACKGROUND label, if not, then it is set to 1.
            if all_classes:
                with open(label, 'r') as f:
                    labels = []
                    for line in f:
                        label = label_dict[line.strip()]
                        labels.append(label)
                labels_out = [0] * len(classes)
                for label in labels:
                    labels_out[label] = 1
                labels_np.append(np.array(labels_out))
            else:
                with open(label, 'r') as f:
                    for line in f:
                        if 'JUMP' in line:
                            label_class = "JUMP"
                            break
                        else:
                            label_class = "BACKGROUND"
                labels_np.append(label_dict[label_class])
            i += 1
            pbar.update(1)
        pbar.close()
        # Convert list of numpy arrays to a single numpy array
        data_np = np.stack(data_np, axis=0)
        data_np = data_np.astype(np.float32)
        labels_np = np.asarray(labels_np)
        #labels_np = np.stack(labels_np, axis=0)
        #labels_np = labels_np.astype(np.float32)
        print('Data shape:', data_np.shape)
        print('Labels shape:', labels_np.shape)
        data_size = (data_np.shape[1], data_np.shape[2])
        
        # Data is now a numpy array of shape (num_files, 1, 128, 87)
        # Data should be of shape (num_files, 128, 87)
        
        #data_np = np.squeeze(data_np, axis=1)
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
        labels_np = np.load(NPLABELPATH, allow_pickle=True)
        
        labels_np = labels_np
        data_np = data_np.astype(np.float32)
        return data_np, labels_np, data_size
import os
import librosa
import soundfile as sf
from tqdm import tqdm


####################
AUDIO_SOURCE_PATH = 'data/measurment-2/raw/230428-006.wav'
AUDIO_SOURCE_LABEL_PATH = 'data/measurment-2/labels/230428-006.txt'
AUDIO_SOURCE_CHANNELS = [0]

AUDIO_OUTPUT_CLIP_PATH = 'data/measurment-2/clips/wav'
AUDIO_OUTPUT_LABEL_PATH = 'data/measurment-2/clips/txt'
AUDIO_OUTPUT_FORMAT = 'wav'
AUDIO_OUTPUT_SUBTYPE = 'PCM_16'

SAMPLE_RATE = 48000

CLIP_LENGTH = 2.0
CLIP_OVERLAP = 1 / 3
####################


# Load the audio file
def load_audio_file(file_path, channels=AUDIO_SOURCE_CHANNELS, sr=SAMPLE_RATE):
    audio, sr = librosa.load(file_path, mono=False, sr=sr)
    audio = audio[channels,:]
    
    return audio

# Read the label file
def read_label_file(file_path):
    labels = list()
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line:
                if line.startswith('#'):
                    continue
                
                if line.split('\t').__len__() != 3:
                    continue
                
                [start, end, label] = line.split('\t')
                labels.append({
                    'start': float(start),
                    'end': float(end),
                    'label': label,
                })
    
    return labels

# Search for labels in a clip
def search_labels_for_clip(labels, clip_start, clip_end):
    clip_labels = list()
    
    for label in labels:
        # Check if the clip is inside the label
        if label['start'] <= clip_start and clip_end <= label['end']:
            clip_labels.append(label)
        
        # Check if the label is inside the clip  
        elif clip_start <= label['start'] and label['end'] <= clip_end:
            clip_labels.append(label)
            
        # Check if the label starts inside the clip
        elif clip_start <= label['start'] and label['start'] <= clip_end:
            clip_labels.append(label)
    
        # Check if the label ends inside the clip
        elif clip_start <= label['end'] and label['end'] <= clip_end:
            clip_labels.append(label)
        
        # No match
        else:
            pass
    
    return clip_labels


# Load the audio file
audio = load_audio_file(AUDIO_SOURCE_PATH)
length = audio.shape[1] / SAMPLE_RATE

# Read the label file
labels = read_label_file(AUDIO_SOURCE_LABEL_PATH)
labels.append({'start': 0.0, 'end': float(length), 'label': 'BACKGROUND'})

# Get unique labels
all_labels = list(set([label['label'] for label in labels]))
all_labels.sort()

print('Unique labels:')
for label in all_labels:
    print(f' - "{label}"')

# Create the output directories
os.makedirs(AUDIO_OUTPUT_CLIP_PATH, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_LABEL_PATH, exist_ok=True)

# Calculate the clip length and overlap in samples
clip_length = int(CLIP_LENGTH * SAMPLE_RATE)
clip_overlap = int(CLIP_OVERLAP * CLIP_LENGTH * SAMPLE_RATE)

print(f'Starting to generate clips from "{AUDIO_SOURCE_PATH}" with a length of {CLIP_LENGTH} seconds and an overlap of {CLIP_OVERLAP:.1%} per clip.')

# Iterate over the audio channels
for idx, channel in enumerate(audio):
    print(f'Generating clips for channel {idx + 1} of {len(audio)}')
    
    start_sample = 0
    end_sample = clip_length
    
    clip_num = 0
    
    # Iterate over the audio clips
    pbar = tqdm(desc=f'Channel {idx}', total=len(channel) // (clip_length - clip_overlap))
    while end_sample < len(channel):
        # Create the clip
        clip = channel[start_sample:end_sample]
        
        # Create the output file name
        audio_file_name = os.path.basename(AUDIO_SOURCE_PATH).split('.')[0]
        clip_sequence = str(clip_num).zfill(4)
        output_file_name = f'clip-{audio_file_name}-{idx}-{clip_sequence}'
        
        # Create the output clip file path
        output_clip_file_name = f'{output_file_name}.{AUDIO_OUTPUT_FORMAT}'
        output_clip_file_path = os.path.join(AUDIO_OUTPUT_CLIP_PATH, output_clip_file_name)
        
        # Save the clip
        sf.write(output_clip_file_path, clip.T, SAMPLE_RATE, subtype=AUDIO_OUTPUT_SUBTYPE, format=AUDIO_OUTPUT_FORMAT)
        
        # Create the output label file path
        output_label_file_name = f'{output_file_name}.txt'
        output_label_file_path = os.path.join(AUDIO_OUTPUT_LABEL_PATH, output_label_file_name)
        
        # Search for labels in the clip
        clip_labels = search_labels_for_clip(labels, start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE)
        
        # Save the clip labels
        with open(output_label_file_path, 'w') as f:
            for clip_label in clip_labels:
                f.write(clip_label['label'] + '\n')
        
        # Update the sample positions
        start_sample += clip_length - clip_overlap
        end_sample += clip_length - clip_overlap
        
        # Update the clip number
        clip_num += 1
        pbar.update(1)
        
    pbar.close()

print('Finished generating clips.')

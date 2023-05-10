import os
import librosa
import soundfile as sf


####################
AUDIO_SOURCE_PATH = 'data/measurment-2/raw/230428-003.wav'
AUDIO_SOURCE_LABEL_PATH = 'data/measurment-2/labels/230428-003.txt'
AUDIO_SOURCE_CHANNELS = [0]

AUDIO_OUTPUT_CLIP_PATH = 'data/measurment-2/jumps/'
AUDIO_OUTPUT_FORMAT = 'wav'
AUDIO_OUTPUT_SUBTYPE = 'PCM_16'

SAMPLE_RATE = 48000

CLIP_PADDING = False # If true, the clips will be padded to the CLIP_LENGTH
CLIP_LENGTH = 20.0
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


# Load the audio file
audio = load_audio_file(AUDIO_SOURCE_PATH)
length = audio.shape[1] / SAMPLE_RATE

# Read the label file
labels = read_label_file(AUDIO_SOURCE_LABEL_PATH)

# Create the output directories
os.makedirs(AUDIO_OUTPUT_CLIP_PATH, exist_ok=True)

# Filter out the jumps
jumps = list(filter(lambda label: label['label'] == 'JUMP', labels))
print(f'Found {len(jumps)} jumps')

print('Generating jump audio files...')

# Iterate over the jumps
for idx, jump in enumerate(jumps):
    print(f'Generating jump audio file {idx + 1} of {len(jumps)}')
    
    if CLIP_PADDING:
        clip_start = jump['start'] - CLIP_LENGTH / 2
        clip_end = jump['start'] + CLIP_LENGTH / 2
    
    else:
        clip_start = jump['start']
        clip_end = jump['end']
    
    clip_start = int(clip_start * SAMPLE_RATE)
    clip_end = int(clip_end * SAMPLE_RATE)
    
    # Check if the clip is inside the audio
    if clip_start < 0:
        clip_start = 0
        
    if clip_end > audio.shape[1]:
        clip_end = audio.shape[1]
        
    # Cut the clip
    clip = audio[:,clip_start:clip_end]
    
    if clip.shape[0] == 1:
        clip = clip[0]
    
    # Create the output file name
    audio_file_name = os.path.basename(AUDIO_SOURCE_PATH).split('.')[0]
    clip_sequence = idx + 1
    pad_str = '-pad' if CLIP_PADDING else ''
    output_file_name = f'{audio_file_name}-jump{pad_str}-{clip_sequence}'
    
    # Create the output clip file path
    output_clip_file_name = f'{output_file_name}.{AUDIO_OUTPUT_FORMAT}'
    output_clip_file_path = os.path.join(AUDIO_OUTPUT_CLIP_PATH, output_clip_file_name)
    
    # Save the clip
    sf.write(output_clip_file_path, clip.T, SAMPLE_RATE, subtype=AUDIO_OUTPUT_SUBTYPE, format=AUDIO_OUTPUT_FORMAT)
    
print('Done!')
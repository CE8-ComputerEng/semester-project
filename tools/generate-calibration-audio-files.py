import sys
sys.path.append('../')
import os
import librosa
import soundfile as sf


# Configuration variables
SAMPLE_RATE = 48000
AUDIO_SUBTYPE = 'PCM_16'
AUDIO_FORMAT = 'WAV'
SOURCE_FOLDER_PATH = 'data/measurment-1/raw'
OUTPUT_FOLDER_PATH = 'data/measurment-1/calibrations'
CALIBRATION_TIMESTAMP = {
    '230320-010.wav': {
        'hydrophone': 'A',
        'channel': 0,
        'calibrations': {
            '0dB': [1, 14],
            '20dB': [17, 29],
            '30db': [31, 44],
        }
    },
    '230320-011.wav': {
        'hydrophone': 'B',
        'channel': 1,
        'calibrations': {
            '0dB': [1, 15],
            '20dB': [17, 30],
            '30db': [33, 45],
        }
    },
}


def load_audio_file(file_path, mono=False, sr=SAMPLE_RATE):
    return librosa.load(file_path, mono=mono, sr=sr)


def save_audio_file(file_path, audio, sr, subtype=AUDIO_SUBTYPE, format=AUDIO_FORMAT):
    sf.write(file_path, audio.T, sr, subtype=subtype, format=format)
    

def generate_calibration_audio(audio, sr, channel, start, end):
    calibration_start = int(start * sr)
    calibration_end = int(end * sr)
    
    calibration_audio = audio[channel,calibration_start:calibration_end]
    
    return calibration_audio


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    
    for file_name, calibration_object in CALIBRATION_TIMESTAMP.items():
        print(f'Generating jump audio files for {file_name}')
        
        audio_path = os.path.join(SOURCE_FOLDER_PATH, file_name)
        audio, sr = load_audio_file(audio_path)
        
        hydrophone = calibration_object['hydrophone']
        channel = calibration_object['channel']
        
        for calibration_name, calibration_timestamps in calibration_object['calibrations'].items():
            calibration_audio = generate_calibration_audio(audio, sr, channel, calibration_timestamps[0], calibration_timestamps[1])
            
            calibration_file_name = file_name.replace('.wav', f'-calibration-{hydrophone}-{calibration_name}.wav')
            calibration_file_path = os.path.join(OUTPUT_FOLDER_PATH, calibration_file_name)
            
            save_audio_file(calibration_file_path, calibration_audio, sr)
            
            print(f'Generated {calibration_file_name}')
            
        
    print('Done')
import os
import librosa
import soundfile as sf


# Configuration variables
SAMPLE_RATE = 48000
JUMP_DURATION = 20.0
AUDIO_SUBTYPE = 'PCM_16'
AUDIO_FORMAT = 'WAV'
SOURCE_FOLDER_PATH = 'data/measurment-1/raw'
OUTPUT_FOLDER_PATH = 'data/measurment-1/jumps'
JUMPS_TIMESTAMP = {
    '230320-008.wav': [
        48.5,
        144.5,
        243.7,
    ],
    '230320-009.wav': [
        48.7,
        135.3,
    ],
}


def load_audio_file(file_path, mono=False, sr=SAMPLE_RATE):
    return librosa.load(file_path, mono=mono, sr=sr)


def save_audio_file(file_path, audio, sr, subtype=AUDIO_SUBTYPE, format=AUDIO_FORMAT):
    sf.write(file_path, audio.T, sr, subtype=subtype, format=format)
    

def generate_jump_audio(audio, sr, jump_position, jump_duration=JUMP_DURATION):
    jump_start = int((jump_position - jump_duration / 2) * sr)
    jump_end = int((jump_position + jump_duration / 2) * sr)
    
    jump_audio = audio[:,jump_start:jump_end]
    
    return jump_audio


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    
    for file_name, jump_positions in JUMPS_TIMESTAMP.items():
        print(f'Generating jump audio files for {file_name}')
        
        audio_path = os.path.join(SOURCE_FOLDER_PATH, file_name)
        audio, sr = load_audio_file(audio_path)
        
        for i, jump_position in enumerate(jump_positions):
            print(f'Generating jump audio file {i + 1} of {len(jump_positions)}')
            
            jump_audio = generate_jump_audio(audio, sr, jump_position)
            output_audio_path = os.path.join(OUTPUT_FOLDER_PATH, file_name.replace('.wav', f'-jump-{i+1}.wav'))
            
            save_audio_file(output_audio_path, jump_audio, sr)
            
            print(f'Jump audio file saved to {output_audio_path}')
            
    print('Done')
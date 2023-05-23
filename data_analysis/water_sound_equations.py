import sys
sys.path.append('../')
import echopype
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

REF_EFF = 20e-5 # Pa
CAL_SPL = 128 # dB


def calculate_calibration_gain(calibration_path, ref_eff=REF_EFF, cal_spl=CAL_SPL):
    print(f'Calibration file: {calibration_path}')
    
    calibration, sr = librosa.load(calibration_path, sr=None, mono=True)
    length = calibration.shape[0] / sr
    print(f'Calibration length: {length:.2f} seconds')
    
    calibration_rms = np.sqrt(np.mean(calibration**2))
    calibration_spl = 20 * np.log10(calibration_rms / ref_eff)
    print(f'Calibration SPL: {calibration_spl:.2f} dB')

    gain_db = calibration_spl - cal_spl
    calibration_gain = 10**(gain_db / 20)
    print(f'Calibration gain: {gain_db:.2f} dB ({calibration_gain})', end='\n\n')
    
    return calibration_gain




CALIBRATION_BASEPATH = 'data/measurment-1/calibrations/'
CALIBRATION_FILE_A = '230320-010-calibration-A-0dB.wav'
CALIBRATION_FILE_B = '230320-011-calibration-B-0dB.wav'

calibration_path_A = os.path.join(CALIBRATION_BASEPATH, CALIBRATION_FILE_A)
calibration_path_B = os.path.join(CALIBRATION_BASEPATH, CALIBRATION_FILE_B)

AUDIO_BASEPATH = 'data/measurment-1/jumps/'
AUDIO_FILE = '230320-009-jump-1.wav'
#AUDIO_BASEPATH = 'data/measurment-2/raw/'
audio_path = os.path.join(AUDIO_BASEPATH, AUDIO_FILE)

audio, sr = librosa.load(audio_path, sr=None, mono=False)

calibration_gain_A = calculate_calibration_gain(calibration_path_A)
calibration_gain_B = calculate_calibration_gain(calibration_path_B)

NFFT = 1024
HOP_LENGTH = 512
TOP_DB = 80
# , win_length= NFFT//2 
spectrogram = librosa.stft(audio, n_fft=NFFT, hop_length=HOP_LENGTH)
spectrogram_dB = librosa.amplitude_to_db(np.abs(spectrogram), top_db=TOP_DB, ref=1)
# NOTE: what ref can be used here? (1, np.max, 2e-5, or what)

spectrogram_dB_A = librosa.amplitude_to_db(np.abs(spectrogram[0]), top_db=TOP_DB, ref=1) - 20 * np.log10(calibration_gain_A)
spectrogram_dB_B = librosa.amplitude_to_db(np.abs(spectrogram[1]), top_db=TOP_DB, ref=1) - 20 * np.log10(calibration_gain_B)

min_dB_A, max_dB_A = np.min(spectrogram_dB_A), np.max(spectrogram_dB_A)
min_dB_B, max_dB_B = np.min(spectrogram_dB_B), np.max(spectrogram_dB_B)

print(f'Hydrophone A - Min: {min_dB_A:.2f} dB, Max: {max_dB_A:.2f} dB')
print(f'Hydrophone B - Min: {min_dB_B:.2f} dB, Max: {max_dB_B:.2f} dB')

plt.figure(figsize=(15, 10))

# cmap should be in same range for both plots
cmap = 'inferno'
cmap_min = np.max([min_dB_A, min_dB_B])
cmap_max = np.max([max_dB_A, max_dB_B])

"""plt.subplot(2, 1, 1)
im = librosa.display.specshow(spectrogram_dB_A, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap='inferno', vmin=cmap_min, vmax=cmap_max)
plt.colorbar(im, format='%+2.0f dB')
plt.title(f'Linear Spectrogram - Hydrophone A - {AUDIO_FILE} - Calibration Gain: {20 * np.log10(calibration_gain_A):.2f} dB')


plt.subplot(2, 1, 2)
im = librosa.display.specshow(spectrogram_dB_B, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear', cmap='inferno', vmin=cmap_min, vmax=cmap_max)
plt.colorbar(im, format='%+2.0f dB')
plt.title(f'Linear Spectrogram - Hydrophone B - {AUDIO_FILE} - Calibration Gain: {20 * np.log10(calibration_gain_B):.2f} dB')

plt.tight_layout()

plt.savefig(f'graphs/spectrogram-{AUDIO_FILE}.png', dpi=300)
plt.show()"""



# First, get paek dB values for each hydrophone
peak_spl_A = np.max(spectrogram_dB_A)
peak_spl_B = np.max(spectrogram_dB_B)
print(peak_spl_A)
# Then, get average dB values for each hydrophone
avg_spl_A = np.mean(spectrogram_dB_A)
print(avg_spl_A)
avg_spl_B = np.mean(spectrogram_dB_B)

# Compute the speed of sound
c = echopype.utils.uwa.calc_sound_speed(temperature=5, salinity=35, pressure=0)
print(c)
# Compute abosoption of sound in water
absorption = echopype.utils.uwa.calc_absorption(frequency=20000, temperature=5, salinity=35, pressure=0, formula_source = "FG")*1000
print(absorption)
# Reduce the peak dB values by the absorption until we reach the average dB values
meter = 0
while peak_spl_A > avg_spl_A:
    peak_spl_A -= absorption
    meter += 1
print(f'Hydrophone A: {meter} m')
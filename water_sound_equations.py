

# First, get paek dB values for each hydrophone
peak_spl_A = np.max(audio_spl_A)
peak_spl_B = np.max(audio_spl_B)
print(peak_spl_A)
# Then, get average dB values for each hydrophone
avg_spl_A = np.mean(audio_spl_A)
print(avg_spl_A)
avg_spl_B = np.mean(audio_spl_B)

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
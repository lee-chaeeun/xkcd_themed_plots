import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.font_manager as font_manager
import sounddevice as sd

# Define constants

# Sampling frequency (Hz)
Fs = 44100  
# Duration of the waveform (seconds)
duration = 2  
csfont = {'fontname':'Comic Sans MS'}
font = font_manager.FontProperties(family='Comic Sans MS', style='normal')

#function to map notes to frequency. 
def note_to_frequency(note):
    a4_frequency = 440.0
    note_mapping = {
        'C': -9, 'C#': -8, 'Db': -8, 'D': -7, 'D#': -6, 'Eb': -6,
        'E': -5, 'Fb': -5, 'E#': -4, 'F': -4, 'F#': -3, 'Gb': -3,
        'G': -2, 'G#': -1, 'Ab': -1, 'A': 0, 'A#': 1, 'Bb': 1,
        'B': 2, 'Cb': 2, 'B#': 3,
    }
    
    base_note = note[:-1]
    octave = int(note[-1])
    half_steps_away = (octave - 4) * 12 + note_mapping[base_note]
    frequency = a4_frequency * 2 ** (half_steps_away / 12.0)
    
    return round(frequency, 2)

def generate_chord():
    # define time 
    t = np.linspace(0, duration, int(Fs * duration), endpoint=False)

    # define notes, in this case emaj13 
    names = ["E4", "G#4", "B4", 'D#5', 'F#5', 'A#5', 'C#6', "E6", 'G6']
    frequencies = [note_to_frequency(name) for name in names]

    # chose random amplitude
    amplitudes = [0.7 for _ in names]

    # characteristic \in You
    eigenschappen = ['enthousiasm', 'generosity', 'rechtvaardig', 'high EQ', 'funny', 'A baller', 'brave', 'epic', 'what a 9.81m/s']

    # assign rainbow colors to frequencies for fun
    colors = [hsv_to_rgb([i / (len(frequencies) - 1.3), 1, 0.9]) for i in range(len(frequencies))]

    # create waveform based on frequencies & random amplitude 
    waveform = np.array([amplitude * np.sin(2 * np.pi * frequency * t) for frequency, amplitude in zip(frequencies, amplitudes)])

    return t, waveform, frequencies, colors, names, eigenschappen

t, waveform, frequencies, colors, names, eigenschappen = generate_chord()
combined_waveform = np.sum(waveform, axis=0)

#FFT Transform to create frequency domain waveform
freq_domain = np.fft.fftfreq(len(t), 1/Fs)
spectrum = np.fft.fft(waveform)

# Play the chord, volgens mij will sound ugly using this library just a warning... 
sd.play(combined_waveform, Fs)
sd.wait()

# Time Domain Plot
plt.figure(figsize=(10, 6))
with plt.xkcd():
    plt.plot(t, combined_waveform)
    plt.title('Sound of You (Time Domain)', **csfont)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

# Frequency Domain Plot
plt.figure(figsize=(10, 6))
with plt.xkcd():
    for i, (freq, color, name, eigenschap) in enumerate(zip(frequencies, colors, names, eigenschappen)):
        plt.plot(freq_domain, np.abs(spectrum[i]), label=f'Note {name} ({freq} Hz)', color=color)
        plt.text(freq, np.abs(spectrum[i])[np.argmax(np.abs(spectrum[i]))], f'{eigenschap}', color=color, fontsize=10, **csfont)
    plt.xlim(min(frequencies) - 50, max(frequencies) + 50)

plt.title('Sound of You (Frequency Domain)', **csfont)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend(bbox_to_anchor=(1.02, 0.1), loc='lower left', borderaxespad=0, prop=font)
plt.grid(True)
plt.tight_layout()

# Show both plots
plt.show()

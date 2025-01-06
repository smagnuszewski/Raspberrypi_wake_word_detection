import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(duration=2, fs=48000, file_prefix="raspi"):
    """
    Records audio for a given duration and sample rate, saving it with an iterative filename.
    """
    # Ensure the 'recordings' directory exists
    if not os.path.exists('ww'):
        os.makedirs('ww')

    # Get the next file number
    files = os.listdir('ww')
    count = len([f for f in files if f.startswith(file_prefix) and f.endswith('.wav')])

    # Record the audio
    print(f"Recording {file_prefix}_{count+1}.wav")
    input("press Enter")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Save the recording
    write(f'ww/{file_prefix}_{count+1}.wav', fs, recording)
    print(f"Saved {file_prefix}_{count+1}.wav")

# Example usage: Record 5 samples
for _ in range(50):
    record_audio()

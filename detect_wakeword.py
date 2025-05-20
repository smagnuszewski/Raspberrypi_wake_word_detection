import pyaudio
import numpy as np
import torch
import time
import librosa
from train_model import RNNModel  # replace with your actual model import

def extract_mfcc(file_path, n_mfcc=13, n_fft=1024, hop_length=512, sr=16000):
    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)  # Load with the specified sampling rate
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    # Mean normalization across time
    mfccs = np.mean(mfccs, axis=1)
    return mfccs

model=RNNModel(13,64,1,1)
model.load_state_dict(torch.load('RNN_wake_word.pth'))
model.eval()
# Audio settings
RATE = 16000
CHUNK = 1024
CHANNELS = 1
DEVICE_INDEX = 1  # Set correctly using PyAudio device info

# Model expects about 1 second of audio
BUFFER_SIZE = RATE  # 1 second
FORMAT = pyaudio.paInt16

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=DEVICE_INDEX)

print("* Listening...")

try:
    buffer = bytes()
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        buffer += data

        # If we've collected 1 second of audio:
        if len(buffer) >= BUFFER_SIZE * 2:  # 2 bytes per sample (16-bit)
            audio_np = np.frombuffer(buffer[:BUFFER_SIZE*2], dtype=np.int16).astype(np.float32)
            buffer = buffer[BUFFER_SIZE*2:]  # clear used buffer

            # Normalize and extract MFCC
            audio_np /= np.max(np.abs(audio_np)) + 1e-9  # normalize to [-1, 1]
            mfccs = extract_mfcc(audio_np)  # ensure your function works on numpy arrays
            feature = torch.tensor([mfccs]).float().unsqueeze(0)

            with torch.no_grad():
                output = model(feature)
                prediction = torch.sigmoid(output).item()

            print(f"Prediction score: {prediction:.4f}")
            if prediction > 0.5:
                print("✅ Word detected!")

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

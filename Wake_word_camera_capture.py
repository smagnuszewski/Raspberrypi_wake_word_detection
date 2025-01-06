import librosa
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
import cv2
from picamera2 import Picamera2, Preview
from PIL import Image



class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step
        out = self.fc(out[:, -1, :]).squeeze()
        return out
    
model=RNNModel(13,64,1,1)
model.load_state_dict(torch.load('RNN_wake_word.pth'))

picam2 = Picamera2()
#camera_config=picam2.create_still_configuration(main={"size": (640,480)}, lores={"size": (640,480)}, display="lores")
camera_config=picam2.create_still_configuration(main={"size": (800,800)}, lores={"size": (640,480)})

#camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
#picam2.start_preview(Preview.QTGL)
picam2.start()

sr = 16000  # Sample rate in Hz
duration = 2         
n_mfcc = 13
n_fft = 1024
hop_length = 512

buffer_size = sr*duration

audio_buffer= np.zeros(buffer_size,dtype=np.float32)
buffer_index=0
model.eval()
set_flag=0

def audio_callback(intdata,frames,time,status):
    global audio_buffer
    global set_flag
    #print("Entered")
    if status:
        print("Error: ",status)

    audio_buffer = np.roll(audio_buffer, -len(intdata))
    
    #amplified_data = intdata[:, 0] * 3 #Ensure the amplified data does not exceed the range of float32 to avoid clipping
    #np.clip(amplified_data, -1, 1, out=amplified_data)
    #audio_buffer[-len(intdata):] = amplified_data
    
    audio_buffer[-len(intdata):] = intdata[:, 0]
    #print(len(audio_buffer))
    if(len(audio_buffer)>= buffer_size):
        mfccs = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs = np.mean(mfccs, axis=1)
        feature=torch.tensor([mfccs]).float().unsqueeze(0)
        with torch.no_grad():
            output=model(feature)
            predicted_prob = torch.sigmoid(output).item()
            if(predicted_prob>=0.65):
                set_flag=1
                print(f"{predicted_prob}")
            else:
                set_flag=0

event=0
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, dtype=np.float32):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            if(set_flag==1 and event==0):
                print('entered')
                event=1
                #picam2.start_preview(Preview.QTGL)
                #picam2.start()
                image=picam2.capture_array()
                picam2.capture_file('Image_captured.jpg')
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('Image captured',image_bgr)
                #time.sleep(1)
                cv2.waitKey(1)  # Allows the window to update
                #picam2.stop_preview()
            else:
                set_flag=0
                event=0
            sd.sleep(500)
    except KeyboardInterrupt:
        print("Stopped listening.")
        cv2.destroyAllWindows()
        #picam2.stop_preview()
        picam2.stop()
        picam2.close()
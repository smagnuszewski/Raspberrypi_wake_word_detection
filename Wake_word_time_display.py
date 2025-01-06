import numpy as np
import sounddevice as sd
import librosa
import adafruit_ssd1306
from PIL import Image, ImageDraw, ImageFont
import time
import board
import math
import datetime
import torch.nn as nn
import torch

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

##### SSD1306####################################
i2c = board.I2C()
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C)

image = Image.new("1", (oled.width, oled.height))

# Get drawing object to draw on image.
draw = ImageDraw.Draw(image)

# Draw a black filled box
draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)

padding = -2
top = padding
bottom = oled.height-padding
x = 0

# font = ImageFont.load_default()
font = ImageFont.truetype('PixelOperator.ttf', 16)

xcenter=25
ycenter=25
radius=25

def map_range(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

def display_clock(hrs,mins,seconds):
  global draw
  draw.text((20,0),'12',font=font, fill=255)     # Clock hour positions
  draw.text((43,19),'3',font=font, fill=255)
  draw.text((19,38),'6',font=font, fill=255)
  draw.text((0,19),'9',font=font, fill=255)
  
  draw.circle((xcenter,ycenter),radius,fill=None,outline=255,width=1)
  
  Sangle=seconds*6       #Second angle
  Mangle=mins*6         #Minute angle
  
  if hrs>=0 and hrs<=12:
    Hangle=30*hrs          #Hour angle
  elif hrs>12:             # Hour format is 24
    Hangle=(hrs-12)*30
  Hangle=Hangle+map_range(Mangle,0,60*6,0,30)
  #print(f'Hangle:{Hangle}')
  # Obtaining the (x,y) coordinates of the Second hand
  shift_sec_x=0.8*radius*math.sin(math.radians(Sangle)) # 0.8 is the ratio of the length
  shift_sec_y=0.8*radius*math.cos(math.radians(Sangle))
  #plotting the second hand
  draw.line(((xcenter,ycenter),((round(xcenter+shift_sec_x),round(ycenter-shift_sec_y)))),width=1,fill=255)
  # Obtaining the (x,y) coordinates of the Minute hand
  shift_min_x=radius*math.sin(math.radians(Mangle))
  shift_min_y=radius*math.cos(math.radians(Mangle))
  #plotting the Minute hand
  draw.line(((xcenter,ycenter),((round(xcenter+shift_min_x),round(ycenter-shift_min_y)))),width=2,fill=255)
  #Obtaining the (x,y) coordinated of hour hand
  shift_hour_x=0.6*radius*math.sin(math.radians(Hangle))
  shift_hour_y=0.6*radius*math.cos(math.radians(Hangle))
  #plotting the hour hand
  draw.line(((xcenter,ycenter),((round(xcenter+shift_hour_x),round(ycenter-shift_hour_y)))),width=2,fill=255)
  draw.text((55,15),str(hrs)+':'+str(mins)+':'+str(seconds),font=ImageFont.truetype('PixelOperator.ttf', 22),fill=255)

  return draw

oled.fill(0)
oled.show()
######################################################

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
start_time=datetime.datetime.now()

def audio_callback(intdata,frames,time,status):
    global audio_buffer
    global start_time
    global set_flag
    #print("Entered")
    if status:
        print("Error: ",status)

    audio_buffer = np.roll(audio_buffer, -len(intdata))
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
                start_time=datetime.datetime.now()
                print(f"{predicted_prob}")
            else:
                set_flag=0
                
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, dtype=np.float32):
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            oled.fill(0)
            #oled.show()
            # Draw a black filled box to clear the image
            draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)
            curr_time=datetime.datetime.now()
            print((curr_time-start_time).total_seconds())
            if(set_flag==1 or ((curr_time-start_time).total_seconds()<=5)):
                #print("disp")
                hours=curr_time.hour
                minutes=curr_time.minute
                seconds=curr_time.second
                draw=display_clock(hours,minutes,seconds)
                oled.image(image)
                
                #time.sleep(0.01)
                #sd.sleep(100)
            else:
                set_flag=0
            oled.show()
            sd.sleep(500)
    except KeyboardInterrupt:
        print("Stopped listening.")
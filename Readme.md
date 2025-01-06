# Wake word Detection on Raspberry Pi

![Image](/Images/Cover%20Image.jpg)

## Introduction
In this project, I have used a RNN model to deploy a wake word detection model on the Raspberry Pi 5. Two separate application is developed with the help of a wake word model:
1. Display Time on wake Word: Uses 128 x 64 pixel SSD1306 OLED display to show current time for 5 seconds when the wake word is detected.
2. Image Capture on OLED display: Captures and image from the Raspberry Pi Camera Module V3 and stores them in local drive when the wake word is detected.

More details can be found here: [Hackster.io Project](https://www.hackster.io/shubhamsantosh99/image-capture-on-wake-word-c99a42)

## Folder Structure:
`/ww_16` audio samples of wake word 'raspi' captured from the microphone at 16 KHz with a 2 second sample duration.

`/nww_16` Audio samples of non-wake words which inculdes random words captured from the microphone at 16 KHz with a 2 second sample duration.

`get_audio_sample.py` Captures audio samples required for wake word detection model.

`wake_word_detection_main.ipynb` Python notebook file for preprocessing, training the wake word model and performing live inferencing.

`wake_word_time_display.py` Source code to run project 1.

`Wake_word_camera_capture.py` Source code to run project 2.


## Steps to run the wake word detection model

1. (Optional)To run custom wake word detection model, use `get_audio_sample.py`. And store them in \ww_16 directory. Additionally, capture samples of other words and store them in \nww_16 directory.
   
2. Generate 'RNN_wake_word.pth':
   
    a. Run the cells on the `wake_word_detection_main.ipynb` to generate the RNN model 'RNN_wake_word.pth'.

    b. Adjust the epochs and learning rate accordingly to fit the model. Use the cell that performs live inferenceing to check if the model is working perfectly.

3. Execute Project-1(Time display):

    a. Ensure that the OLED display and the INMP441 microphone is setup correctly.

    b. Run the code `wake_word_time_display.py`
    
    Demo: 

    [![Project-1](https://img.youtube.com/vi/TjZlh7XeYAc/1.jpg)](https://www.youtube.com/watch?v=TjZlh7XeYAc)

4. Execute Project-2(Image Capture):
   a. Make sure that the Raspberry Pi Camera Module V3 is setup correctly.

   b. Run the code `Wake_word_camera_capture.py`.

    Demo

    [![Project-2](https://img.youtube.com/vi/KbtO2zOsly4/1.jpg)](https://www.youtube.com/watch?v=KbtO2zOsly4)




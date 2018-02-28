This repository contains the code for Advanced Driver Assistance System using Traffic signs & Driver EEG data.

Summary of the Project:

* We used EEG signal to detect the drowsiness of driver and also parallely detects the traffic signs from the dash board camera.

* belgium traffic sign dataset is used for deep learning model training - http://btsd.ethz.ch/shareddata/.

* EEG brain wave data from Kaggle is used for detecting the driver drowsiness detection - https://www.kaggle.com/wanghaohan/eeg-brain-wave-for-confusion.

* For the traffic sign board recognition we resized all the images into 32×32×3 and used three fully connected layers with 200,100,62 number neurons in each layer.

* Linear SVM is used to classify EEG Signal data for driver drowsiness detection.


Description of the python file:

1. EEG_python_out.py - EEG signal data classification

2. sign_model_3_out.py - Traffic Sign board recognition

3. combined_out_main.py - Final classification based on the EEG data and Traffic sign board recognition.



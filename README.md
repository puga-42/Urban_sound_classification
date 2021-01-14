# Urban_sound_classification

# The Data
The data was taken from the UrbanSounds8k dataset. It is composed of 8,723 sound snippets taken from 1297 unique sounds of 10 classes:

Class 0:    Air Conditioners
![alt text](img/0_airconditioner.png "Title")


Class 1:    Car Horns
![alt text](img/1_car.png "Title")

Class 2:    Children Playing
![alt text](img/2_children.png "Title")

Class 3:    Dog Bark
![alt text](img/3_dog.png "Title")

Class 4:    Drilling
![alt text](img/4_drilling.png "Title")

Class 5:    Engine Idling
![alt text](img/5_engine.png "Title")

Class 6:    Gunshot
![alt text](img/6_gunshot.png "Title")

Class 7:    Jackhammer
![alt text](img/7_jackhammer.png "Title")

Class 8:    Siren
![alt text](img/8_siren.png "Title")

Class 9:    Street Music
![alt text](img/9_street_music.png "Title")




# Features Extracted

# Models Used
- Random Forest

# Feature performance with Random Forest
A random forest trained only on MFCCs achieved an accuracy of 64.84%
This vastly outpreforms all other features, with the next closest features, spectral and chroma, achieving 45% accuracy.

# Where does the model trained on MFCCs fail?
This model is clearly having a rough time differentiating the prolonged, droning sounds of air conditioners, idle engines, car horns (probably accompanined by the drone of traffic), drilling, and jackhammers. Let's see if the introduction of some other features can improve this model!

# Trained on spectral as well
This model does much worse. It still has the same struggles as the MFCCs but now it is doing worse on the sounds like gunshots and dog barks that the other model was doing great at identifying.

# Let's investigate some actual sounds


It sounds like the air conditioner is usually more of a constant drone than the idle engine or the jackhammer. Maybe if we implemented a feature that can pick up on the key or sustained frequency of the sounds we would get a better answer. We are kind of doing this with chroma_stft, but let's try tonnetz and see if we get anything different.

# Final Model
## 50 MFCCs, ZCR, Chroma STFT, Spectral Contrast

![alt text](img/cm_zcr_chroma_spectral_mfcc.png "Title")

Ultimately we achieved an accuracy of 70.2%. I didn't expect the MFCC's to perform so well, but I guess there's a reason they've been one of the most used features in audio classification over the past few decades. Despite the improvement, the model has a very hard time predicting white noises like air conditioners and engines idling.

-----

## Why did the Mel-Frequency Cepstral Coefficients Perform so Well?

### I have one main hypothesis for the MFCC's success:
MFCC's are used for detecting speech because they are good at seperating signals produced from vocal tract movements from signals produced from glottal pulses. It's possible that some background noises were classified as glottal pulses and were seperated from the rest of the signal. 

# Next Steps

It's clear that audio signals with dense frequency plots are giving the model the most trouble. I want to try and tackle this problem in two ways:

1 - Using a Neural Net to predict classes. Training over 10 folds will take some time but could yield interesting results.


2 - Implement a noise reduction method. Sounds like air conditioners have low variability and a feature could be designed to compare the energy of pre and post noise cancelled signals.


# Urban_sound_classification

# The Data
The data was taken from the UrbanSounds8k dataset. It is composed of 8,723 sound snippets taken from 1297 unique sounds of 10 classes:

0:
![alt text](img/0_airconditioner.png "Title")


1:
![alt text](img/1_car.png "Title")

2:
![alt text](img/2_children.png "Title")

3:
![alt text](img/3_dog.png "Title")

4:
![alt text](img/4_drilling.png "Title")

5:
![alt text](img/5_engine.png "Title")

6:
![alt text](img/6_gunshot.png "Title")

7:
![alt text](img/7_jackhammer.png "Title")

8:
![alt text](img/8_siren.png "Title")

9:
![alt text](img/9_street_music.png "Title")



## Example Sounds from Each Class


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

The last model was so close to 70% - I had to find one a feature to push it over the top.


# Next Steps

It's clear that audio signals with dense frequency plots are giving the model the most trouble. I want to try and tackle this problem in two ways:
1 - Using a Neural Net to predict classes. Training over 10 folds will take some time but could yield interesting results.


2 - Implement a noise reduction method. Sounds like air conditioners have low variability and a feature could be designed to compare the energy of pre and post noise cancelled signals.


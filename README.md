# Urban_sound_classification

# The Data

# Features Extracted

# Models Used
- Random Forest
- hopefully NN

# Feature performance with Random Forest
A random forest trained only on MFCCs achieved an accuracy of 64.84%
This vastly outpreforms all other features, with the next closest features, spectral and chroma, achieving 45% accuracy.

# Where does the model trained on MFCCs fail?
This model is clearly having a rough time differentiating the prolonged, droning sounds of air conditioners, idle engines, car horns (probably accompanined by the drone of traffic), drilling, and jackhammers. Let's see if the introduction of some other features can improve this model!

# Trained on spectral as well
This model does much worse. It still has the same struggles as the MFCCs but now it is doing worse on the sounds like gunshots and dog barks that the other model was doing great at identifying.

# Let's investigate some actual sounds


It sounds like the air conditioner is usually more of a constant drone than the idle engine or the jackhammer. Maybe if we implemented a feature that can pick up on the key or sustained frequency of the sounds we would get a better answer. We are kind of doing this with chroma_stft, but let's try tonnetz and see if we get anything different.




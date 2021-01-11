import librosa
import numpy as np
import os
import pickle
from pathlib import Path

class AudioFeature:
    def __init__(self, src_path, fold, label):
        self.src_path = src_path
        self.fold = fold
        self.label = label
        self.y, self.sr = librosa.load(self.src_path, mono=True)
        self.features = None
        self.selected_features = []
        
        
    def concat_features(self, feature):
        """
        Whenever a self.get_features() method is called in this class,
        this function concatenates to the self.features feature vector
        """
        self.features = np.hstack(
            [self.features, feature] if self.features is not None else feature
        )
        
    def extract_mfcc(self, n_mfcc=25):
        mfcc = librosa.feature.mfcc(self.y, sr=self.sr, n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        self.concat_features(mfcc_feature)
    
        
    def extract_rolloff(self):
        
        rolloff = librosa.feature.spectral_rolloff(self.y, self.sr)
        rolloff_mean = rolloff.mean(axis=1).T
        rolloff_std = rolloff.std(axis=1).T
        rolloff_feature = np.hstack([rolloff_mean, rolloff_std])
        self.concat_features(rolloff_feature)
    
    
    def extract_spec_cent(self):
        spec_cent = librosa.feature.spectral_centroid(self.y, self.sr)

        spec_cent_mean = spec_cent.mean(axis=1).T
        spec_cent_std = spec_cent.std(axis=1).T
        spec_cent_feature = np.hstack([spec_cent_mean, spec_cent_std])
        self.concat_features(spec_cent_feature)

    def extract_bw(self):
        bw = librosa.feature.spectral_bandwidth(self.y, self.sr)

        bw_mean = bw.mean(axis=1).T
        bw_std = bw.std(axis=1).T
        bw_feature = np.hstack([bw_mean, bw_std])
        self.concat_features(bw_feature)
        
    def extract_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(self.y, self.sr)

        zcr_mean = zcr.mean(axis=1).T
        zcr_std = zcr.std(axis=1).T
        zcr_feature = np.hstack([zcr_mean, zcr_std])
        self.concat_features(zcr_feature)
        
    def extract_chroma_stft(self):
        stft = np.abs(librosa.stft(self.y))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=self.sr)
        chroma_mean = chroma_stft.mean(axis=1).T
        chroma_std = chroma_stft.std(axis=1).T
        chroma_feature = np.hstack([chroma_mean, chroma_std])
        self.concat_features(chroma_feature)
        
    def extract_spectral_contrast(self, n_bands=3):
        spec_con = librosa.feature.spectral_contrast(y=self.y, sr=self.sr, n_bands=n_bands)
        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        self.concat_features(spec_con_feature)
        
    
    def get_features(self, *feature_list, save_local=True):
        extract_fn = dict(
            mfcc=self.extract_mfcc,
            spec_cent=self.extract_spec_cent,
            rolloff=self.extract_rolloff,
            bw=self.extract_bw,
            zcr=self.extract_zcr,
            spectral=self.extract_spectral_contrast,
            chroma=self.extract_chroma_stft,
        )
        
        for feature in feature_list:
            extract_fn[feature]()

        if save_local:
            self.save_local()
    

    def save_local(self, clean_source=True):
        out_name = self.src_path.split("/")[-1]
        out_name = out_name.replace(".wav", "")

        filename = f"{Path.home()}/Desktop/gal_notes/Capstone/Urban_sound_classification/pkl/fold{self.fold}/{out_name}.pkl"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)

        if clean_source:
            self.y = None
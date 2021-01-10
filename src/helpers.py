import pandas as pd
import numpy as np
import IPython.display as ipd
import os
import librosa
import csv



def make_feature_file():
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()
    file = open('data/c3_features.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
make_feature_file()

def featurize(wav_file, y, sr):
    filename = wav_file
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        
    write_to_file(to_append)
        
def get_features(base_path):
    i = 0
    folds = []
    for i in range(1, 11):
        folds.append('fold' + str(i) + '/')
    
    for fold in folds:
        fold_path = base_path + fold
        path_list = os.listdir(fold_path)
#         path_list = os.listdir(os.path.join(base_path, fold))

        for wav_file in path_list:
            if wav_file != '.DS_Store':
                filename = fold_path + wav_file

                y, sr = librosa.load(filename, res_type='kaiser_fast')
                feat = featurize(filename, y, sr)
                i += 1
                if i%500 == 0:
                    print(i)
            
def write_to_file(row):            
    file = open('data/c3_initial_features.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(row.split())
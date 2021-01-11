from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path


def initialize_audio_features():
    metadata = parse_metadata('../data/UrbanSound8K/metadata/UrbanSound8K.csv')

    audio_features = []

    for line in tqdm(metadata):
        path, fold, label = line[0], line[1], line[2]
        
        

        fn = path.replace(".wav", "")
        transformed_path = f"{Path.home()}/Desktop/gal_notes/Capstone/Urban_sound_classification/pkl/fold{fold}/{fn}.pkl"

        if os.path.isfile(transformed_path):
            # if the file exists as a .pkl already, then load it
            with open(transformed_path, "rb") as f:
                audio = pickle.load(f)
                audio_features.append(audio)
        else:
            # if the file doesn't exist, then extract its features from the source data and save the result
            src_path = f"{Path.home()}/Desktop/gal_notes/Capstone/Urban_sound_classification/data/UrbanSound8K/audio/fold{fold}/{path}"
            # get features
            audio = AudioFeature(src_path, fold, label)
            
            audio.get_features("mfcc", "bw", "rolloff", "spec_cent", 
                            "zcr", "spectral", "chroma")
            
            audio_features.append(audio)
    return audio_features



def parse_metadata(path):
    meta_df = pd.read_csv(path)
    meta_df = meta_df[["slice_file_name", "fold", "class"]]
    meta = zip(meta_df["slice_file_name"], meta_df["fold"], meta_df["class"])

    return meta

def select_features(feat_list, single_audio):
    selected_features = []

    
    if 'mfc' in feat_list:
        selected_features += list(single_audio[0:50])
        
    if 'spec_cent' in feat_list:
        selected_features += list(single_audio[54:56])
        
    if 'rolloff' in feat_list:
        selected_features += list(single_audio[52:54])
        
    if 'bw' in feat_list:
        selected_features += list(single_audio[50:52])
        
    if 'zcr' in feat_list:
        selected_features += list(single_audio[56:58])
        
    if 'spectral' in feat_list:
        selected_features += list(single_audio[58:66])
        
    if 'chroma' in feat_list:
        selected_features += list(single_audio[66:90])
    
    return selected_features
        

def get_model_stats(features_to_test, feature_matrix):
    
    testing_features = []
    
    for single_audio in feature_matrix:
        ## get the selected chosen features in each preloaded feature matrix entry
        single_chosen_features = select_features(features_to_test, single_audio)
        testing_features.append(single_chosen_features)
   
    ## fit the model
    testing_features = np.array(testing_features)
    model = model.Model(testing_features, labels, folds, model_cfg)
    fold_acc, predicted_labels, actual_labels = model.train_kfold()
    
    return fold_acc, predicted_labels, actual_labels

def prediction_df(predicted_labels, actual_labels):
    class_ids = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        4: 'drilling',
        5: 'engine_idling',
        6: 'gun_shot',
        7: 'jackhammer',
        8: 'siren',
        9: 'street_music'
    }

    list_to_df = []
    for i in range(len(predicted_labels)):
        actual_class = ''
        predicted_class = ''
        class_id = 0
        fold = i
    
    
        ## get an array of 0s and 1s where wrong predictions occured
        diff = predicted_labels[i] - actual_labels[i]
        diff[diff != 0] = 1
        ##
        for j in range(len(diff)):
            if diff[j] == 1:
                actual_class = actual_labels[i][j]
                actual_name = class_ids[actual_class]
                predicted_class = predicted_labels[i][j]
                predicted_name = class_ids[predicted_class]
                
                list_to_df.append([actual_class, actual_name,
                                predicted_class, predicted_name,
                                fold])
            

    wrong_preds = pd.DataFrame(list_to_df, columns=['actual_class', 'actual_name',
                                                    'predicted_class', 'predicted_name',
                                                    'fold'])
    return wrong_preds


def visualize_predictions(df, false_negatives=False, false_positives=False):
    ## Visualizing number of classes that weren't predicted
    if false_negatives:
        fig, ax = plt.subplots(figsize=(14,10))
        bars = df[['actual_name', 
                            'fold']].groupby('actual_name').agg('count').sort_values('fold')

        ax.barh(bars.index, bars['fold'])
        ax.set_title('Number of Times Each Class was Mislabeled')
        ax.set_xlabel('Number of False Negatives')
        ax.set_ylabel('Sound Class');
        
    elif false_positives:
        fig, ax = plt.subplots(figsize=(14,10))
        bars = df[['predicted_name', 
                            'fold']].groupby('predicted_name').agg('count').sort_values('fold')

        ax.barh(bars.index, bars['fold'])
        ax.set_title('Number of Times Each Class was Falsely Labeled')
        ax.set_ylabel('Sound Class')
        ax.set_xlabel('Number of False Positives');
import os
import librosa
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from .audio_utils import (wake_word, record_background, adding_noise, 
                          shifting, speed_pitch, feature_extraction)

def remove_silence(audio):
    intervals = librosa.effects.split(audio, top_db=20)
    return np.concatenate([audio[start:end] for start, end in intervals]) if len(intervals) > 0 else audio

def prepare_data(data_path):
    y_vocal, y_background = [], []
    
    for audio_file in os.listdir(data_path):
        if audio_file.endswith('.wav'): 
            file_path = os.path.join(data_path, audio_file)
            y1, sr1 = librosa.load(file_path)
            
            y_back = record_background(y1, sr1)
            y_voice = wake_word(y1, sr1)
            
            # Augmentations
            y_voice_aug = [
                y_voice, 
                shifting(y_voice, sr1), 
                adding_noise(y_voice), 
                speed_pitch(y_voice, sr1)
            ]
            y_back_aug = [
                y_back, 
                shifting(y_back, sr1), 
                adding_noise(y_back), 
                speed_pitch(y_back, sr1)
            ]
            
            # Remove silence and extract features for vocals
            for aug in y_voice_aug:
                clean_aug = remove_silence(aug)
                mfcc = feature_extraction(clean_aug, sr1).flatten()
                y_vocal.append(mfcc)
                
            # Remove silence and extract features for background
            for aug in y_back_aug:
                clean_aug = remove_silence(aug)
                mfcc = feature_extraction(clean_aug, sr1).flatten()
                y_background.append(mfcc)

    y_vocal = torch.stack(y_vocal)
    y_background = torch.stack(y_background)
    
    labels_vocal = torch.zeros(len(y_vocal), 1)
    labels_back = torch.ones(len(y_background), 1)
    
    X = torch.cat((y_background, y_vocal), dim=0)
    y = torch.cat((labels_back, labels_vocal), dim=0)
    
    return X, y

def get_dataloaders(data_path, batch_size=8):
    X, y = prepare_data(data_path)
    
    # Split data (80% Train+Test / 20% Valid)
    X_temp, X_valid, y_temp, y_valid = train_test_split(X, y, test_size=0.2)
    # Split remaining into (75% Train / 25% Test of the 80%) -> 60% Train, 20% Test total
    X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=0.25)

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_valid)}")
    print(f"Test set size: {len(X_test)}")

    train_dl = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    valid_dl = DataLoader(TensorDataset(X_valid, y_valid), shuffle=True, batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(X_test, y_test), shuffle=True, batch_size=batch_size)
    
    return train_dl, valid_dl, test_dl

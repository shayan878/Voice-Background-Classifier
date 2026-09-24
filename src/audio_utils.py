import librosa
import numpy as np
import torch

POWER = 3

def wake_word(y, sr):
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=15)
    S_filter = np.minimum(S_full, S_filter)
    margin_v = 11
    mask1 = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=POWER)
    S_hi = mask1 * S_full
    return librosa.istft(S_hi * phase)

def record_background(y, sr):
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=15)
    S_filter = np.minimum(S_full, S_filter)
    margin_i = 3
    mask2 = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=POWER)
    S_back = mask2 * S_full
    return librosa.istft(S_back * phase)

def adding_noise(audio):
    noise_level = 0.1
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise

def shifting(audio, sample_rate):
    max_shift = 0.2
    shift_samples = int(np.random.uniform(-max_shift * sample_rate, max_shift * sample_rate))
    return np.roll(audio, shift_samples)

def speed_pitch(audio, sample_rate):
    speed_factor = 1.2
    y_pitch = librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=2)
    return librosa.effects.time_stretch(y=y_pitch, rate=speed_factor)

def feature_extraction(y_voice, sr_voice):  
    max_len = 100
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y_voice, sr=sr_voice, n_mfcc=n_mfcc).T
    
    if mfcc.shape[0] > max_len:
        mfcc = mfcc[:max_len, :]
    else:
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
    return torch.Tensor(mfcc)

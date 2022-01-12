import sounddevice as sd
from scipy.signal import istft
from scipy.signal import stft
import librosa
import librosa.display
import midi
import skimage.transform
import numpy as np
import os
import h5py
import time
import matplotlib.pyplot as plt
start_time = time.time()


def make_wave(freq, duration, sample_rate = 22050):
    wave = []
    for i in range(0,duration*sample_rate):
        wave.append(i/((sample_rate/(2*np.pi))/freq))

    wave = np.cos(np.stack(wave))
    return wave

A = make_wave(440, 10)
#sd.play(A)
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f,t,specgram = stft(A,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)


for n,freq in enumerate(specgram):
    print(n,":",abs(freq[0]))


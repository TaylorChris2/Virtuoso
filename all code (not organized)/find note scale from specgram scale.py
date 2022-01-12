import sounddevice as sd
import numpy as np
import librosa
import librosa.display
from sympy import ifft
from matplotlib import mlab
from matplotlib import pyplot as plt
from matplotlib import pyplot
from scipy.signal import istft
from scipy.signal import stft


y, sr = librosa.load("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data/S3 Alg 1 of 4/A1-0001_allegro assai 1 of 4_00086400.wav")
print("orig shape:",y.shape)
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000

f,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)

for thing in f:
    print(thing)


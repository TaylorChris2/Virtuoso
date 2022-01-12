import numpy as np                                       # fast vectors and matrices
import matplotlib.pyplot as plt                          # plotting
from scipy import fft                                    # fast fourier transform

import librosa
from intervaltree import Interval,IntervalTree




X, fs = librosa.load("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data/S3 Alg 1 of 4/A1-0001_allegro assai 1 of 4_00086400.wav")

window_size = 2048  # 2048-sample fourier windows
stride = 512        # 512 samples between windows
wps = fs/float(512) # ~86 windows/second
print("wps:",wps)
Xs = []


index=0
while True:
    if np.abs(fft(X[index*stride:index*stride+window_size])).shape[0]==2048:
        Xs.append(np.abs(fft(X[index*stride:index*stride+window_size])))
    else:
        break
    index+=1

Xs = np.stack(Xs)
print(Xs.shape[0]/wps)

print(np.max(Xs))

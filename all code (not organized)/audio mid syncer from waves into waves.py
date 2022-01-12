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


path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 2"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced"

if not os.path.exists(output+"/wavs"):
    os.makedirs(output+"/wavs")
if not os.path.exists(output+"/midis"):
    os.makedirs(output+"/midis")
factor = 1
good = ""
save_index = 0
for set_ in os.listdir(path):
    for item in os.listdir(path+"/"+set_):
        if item.endswith(".mid"):
            orig_midi = librosa.load(path+"/"+set_+"/"+item)
        elif item.endswith(".wav"):
            orig_wav = librosa.load(path+"/"+set_+"/"+item)

    index = 0
    while True:
        actual_midi = orig_midi[index*8192:]
        actual_wav = orig_wav[index*8192:]
        if actual_wav.shape[0] == 0:
            break
        while good != "y":
            factor = input("enter a factor:")
            midi = skimage.transform.rescale(orig_midi,(float(factor), 1))
            if midi.shape[0] > orig_wav.shape[0]:
                midi = midi[:orig_wav.shape[0]]
            elif midi.shape[0] < orig_wav.shape[0]:
                padding_amt = orig_wav.shape[0]-orig_midi.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in orig_wav:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi = np.stack(padded)
            midi = midi[:8192]
            wav = wav[:8192]
            good = input("did it sound good?:")
            print(midi.shape)
        save_array(wav, output+"/wavs"+str(save_index))
        save_array(midi, output+"/midis"+str(save_index))
        save_index+=1
        index+=1
        




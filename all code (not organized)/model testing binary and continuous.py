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
import scipy
start_time = time.time()

def save_array(array, path):
    while True:
        try:
            if os.path.exists(path):
                os.remove(path)

            hdf5_store = h5py.File(path, "a")
            hdf5_store.create_dataset("all_data", data = array, compression="gzip")
            break
        except:
            pass

def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array

def load_predicted(binary_path, continuous_path):
    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000

    frequency_clip_wav = 512

    array1 = load_array(continuous_path)[0]
    array2 = load_array(binary_path)[0]
    print(np.unique(array2))
    array2 = array2 > 0.5
    array1[array2 == 0] = 0
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.squeeze(array1,axis=2))
    plt.show()


    specgram = np.transpose(np.squeeze(array2,axis=2))

    decoded = []
    for freq in specgram:
        decoded.append(freq)
    for i in range(0,(1025-frequency_clip_wav)):
        decoded.append(np.zeros(specgram.shape[1]))
    decoded = np.stack(decoded)
    decoded = (decoded*100)-100
    decibels = librosa.db_to_amplitude(decoded)
    t,back = istft(decibels,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    back = back*0.1/np.max(back)

    return back

def load_true(path):
    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000

    frequency_clip_wav = 512

    array1 = load_array(path)[0]

    specgram = np.transpose(np.squeeze(array1,axis=2))

    decoded = []
    for freq in specgram:
        decoded.append(freq)
    for i in range(0,(1025-frequency_clip_wav)):
        decoded.append(np.zeros(specgram.shape[1]))
    decoded = np.stack(decoded)
    decoded = (decoded*100)-100
    decibels = librosa.db_to_amplitude(decoded)
    t,back = istft(decibels,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    back = back*0.1/np.max(back)

    return back

continuous_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/test prediction spectrograms/Continuous synced 1/61_true.h5"
binary_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/test prediction spectrograms/Binary 2/11_true.h5"
tru_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/test prediction spectrograms/Continuous synced 2/61_true.h5"

save_folder = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/test prediction waves"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

true = load_true(tru_path)
pred = load_predicted(binary_path, continuous_path)

#sd.play(true,22050)
#cont = input("...")
sd.play(pred,22050)


scipy.io.wavfile.write(save_folder+"/pred.wav",22050,pred)
scipy.io.wavfile.write(save_folder+"/true.wav",22050,true)


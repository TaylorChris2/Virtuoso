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

def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array


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


path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/unsynced"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000

wav_path = path+"/wavs"
midi_path = path+"/midis"

wavs = []
for file in os.listdir(wav_path):
    wavs.append(file)

midis = []
for file in os.listdir(midi_path):
    if "flat" not in file:
        midis.append(file)

flat_midis = []
for file in os.listdir(midi_path):
    if "flat" in file:
        flat_midis.append(file)

if not os.path.exists(output+"/wavs"):
    os.makedirs(output+"/wavs")
if not os.path.exists(output+"/midis"):
    os.makedirs(output+"/midis")


save_index = 0
start = 0

midi_durations = []
for n in range(start, len(wavs)):
    orig_midi = load_array(midi_path+"/"+midis[n])
    orig_midi_flat = load_array(midi_path+"/"+flat_midis[n])
    orig_wav = load_array(wav_path+"/"+wavs[n])
    wav_tick = 0
    midi_tick = 0
    while True:
        good = ""
        actual_midi = orig_midi[midi_tick:]
        actual_midi_flat = orig_midi_flat[midi_tick:]
        actual_wav = orig_wav[wav_tick:]
        print(midi_tick,wav_tick)
        if actual_wav.shape[0] == 0:
            break
        good = float(input("enter a factor:"))
        while good != "y":
            
            real_length = round(2048/float(good))
            print(real_length)
            midi = skimage.transform.rescale(actual_midi,(float(good), 1))
            if midi.shape[0] > actual_wav.shape[0]:
                midi = midi[:actual_wav.shape[0]]
            elif midi.shape[0] < actual_wav.shape[0]:
                padding_amt = actual_wav.shape[0]-midi.shape[0]
                padding = np.zeros((padding_amt,actual_wav.shape[1]))
                padded = []
                for time_ in midi:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi = np.stack(padded)
            reg_midi = midi.copy()
            ## for the flattened version
            midi = skimage.transform.rescale(actual_midi_flat,(float(good), 1))
            if midi.shape[0] > actual_wav.shape[0]:
                midi = midi[:actual_midi_flat.shape[0]]
            elif midi.shape[0] < actual_wav.shape[0]:
                padding_amt = actual_wav.shape[0]-midi.shape[0]
                padding = np.zeros((padding_amt,actual_wav.shape[1]))
                padded = []
                for time_ in midi:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi = np.stack(padded)
            flattened_midi = midi.copy()
            wav = actual_wav[:2048]
            midi = reg_midi[:2048]
            flat_midi = flattened_midi[:2048]
            print(wav.shape,midi.shape)
            ##playing
            converted_back_midi = np.transpose(flat_midi)
            decoded = []
            for freq in converted_back_midi:
                decoded.append(freq)
            for i in range(0,(1025-frequency_clip_midi)):
                decoded.append(np.zeros(converted_back_midi.shape[1]))
            decoded = np.stack(decoded)
            decoded = (decoded*100)-100
            decoded = librosa.db_to_amplitude(decoded)
            t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
            back_midi = back*0.3/np.max(back)

            converted_back_wav = np.transpose(wav)
            decoded = []
            for freq in converted_back_wav:
                decoded.append(freq)
            for i in range(0,(1025-frequency_clip_wav)):
                decoded.append(np.zeros(converted_back_wav.shape[1]))
            decoded = np.stack(decoded)
            decoded = (decoded*100)-100
            decoded = librosa.db_to_amplitude(decoded)
            t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
            back_wav = back*0.1/np.max(back)

            added = back_wav+back_midi
            sd.play(added,5512)
            good = input("did it sound good? (enter next factor for no) (yes for good) (b to go back one) (r to replay):")
            while good == "r":
                speed = int(input("what speed do you wanted it played at?:"))
                sd.play(added,speed)
                good = input("did it sound good? (enter next factor for no) (yes for good) (b to go back one) (r to replay):")
            if good == "b":
                backing = True
                break
            else:
                backing = False
            print(midi.shape)
        print("exited")
        save_array(wav, output+"/wavs/"+str(save_index))
        save_array(midi, output+"/midis/"+str(save_index))
        if not backing:
            save_index+=1
            wav_tick += 2048
            midi_durations.append(real_length)
            midi_tick += real_length
        else:
            save_index-=1
            wav_tick -= 2048
            midi_tick -= midi_durations[-1]
            del midi_durations[-1]
            




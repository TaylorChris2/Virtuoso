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

def load_midi_violin(path):
    note_events = []
    mid = midi.read_midifile(path)
    ##getting only the note data
    for n,track in enumerate(mid):
        note_events.append([])
        for event in track:
            if "NoteOnEvent" in str(event):
                note_events[n].append(event)
            elif "NoteOffEvent" in str(event):
                event.data[1] = 0
                note_events[n].append(event)
                       
    ##deleting empty tracks
    only_notes = []
    for n,track in enumerate(note_events):
        if len(track)>0:
            only_notes.append(track)
            
    ##getting track length
    track_lengths = []
    for n,track in enumerate(only_notes):
        track_lengths.append(0)
        for event in track:
            track_lengths[n] += event.tick
    track_length = max(track_lengths)
    
    ##creating the actual track array and filling with empties
    track_array = []
    for i in range(0,track_length):
        track_array.append([[0.,0.,0.,0.],[1.,1.,1.,1.]])##one four channel list for pitch and one for articulation
    track_array = np.stack(track_array)
    ##filling in the track array with real note data
    for track in only_notes:
        current_tick = 0
        for n,event in enumerate(track):
            current_tick += event.tick
            if event.data[1] == 100:##every note start
                
                for i in range(current_tick,current_tick+track[n+1].tick):
                    for slot in range(0,4):
                        if track_array[i][0][slot] == 0:
                            track_array[i][0][slot] = event.data[0]
                            working_slot = slot
                            break
                for i in range(0,int(track[n+1].tick/4)):
                    track_array[current_tick+i][1][working_slot] = i/int(track[n+1].tick/4)
                    track_array[current_tick+track[n+1].tick-i-1][1][working_slot] = i/int(track[n+1].tick/4)
                    
    return track_array     



def seperate_sets(midis, mels, set_size):
    midi_sets = []
    mel_sets = []
    loop = 0
    current_set = -1
    num_sets = len(midis)
    
    while True:
        if loop % set_size == 0:
            midi_sets.append([])
            mel_sets.append([])
            current_set += 1

        midi_sets[current_set].append(midis[loop])
        mel_sets[current_set].append(mels[loop])
        loop += 1

        if loop >= num_sets:
            break
        
    return midi_sets, mel_sets


def save_data_set(set_, save_path, save_name):
    if os.path.exists(os.path.join(save_path, save_name)+".h5"):
        os.remove(os.path.join(save_path, save_name)+".h5")

    hdf5_store = h5py.File(os.path.join(save_path, save_name)+".h5", "a")
    hdf5_store.create_dataset("all_data", data = set_, compression="gzip")

def split_train_val_test(set_):
    total = len(set_)
    train_end_val_beginning = round(0.7 * total)
    val_end_test_beginning = round(0.85 * total)


    train_images = set_[:train_end_val_beginning]
    val_images = set_[train_end_val_beginning:val_end_test_beginning]
    test_images = set_[val_end_test_beginning:]

    return train_images, val_images, test_images

def make_wave(freq, duration, sample_rate = 22050):
    wave = []
    for i in range(0,int(duration*sample_rate)):
        wave.append(i/((sample_rate/(2*np.pi))/freq))

    wave = np.sin(np.stack(wave))
    return wave

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

def generate_specgrams(output_path, max_pitch):
    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    ov = 250
    for pitch in range(69,max_pitch):
        print(pitch)
        freq = 440*(2**((pitch-69)/12))
        wave = make_wave(freq, 1, sample_rate=22050)
        sd.play(wave)
        cont = input("...")
        f,t,specgram = stft(wave,nfft=N,fs=Fs)
        test = []
        for thing in specgram:
            test.append(thing)
        for thing in specgram:
            test.append(thing)
        #specgram = np.stack(test)
        t,back = istft(specgram,nfft=N,fs=Fs)
        sd.play(back,22050)
        cont = input("...")
        sd.play(back,22050)
        save_array(specgram, output_path+"/"+str(pitch))
        break
    
        

            

save_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/frequencies 2"
if not os.path.exists(save_path):
    os.makedirs(save_path)
generate_specgrams(save_path, 119)


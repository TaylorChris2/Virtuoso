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
    ov = N - Fs // 22050
    #ov = 2025
    for pitch in range(69,max_pitch):
        print(pitch)
        freq = 440*(2**((pitch-69)/12))
        wave = make_wave(freq, 10, sample_rate=22050)
        f,t,specgram = stft(wave,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        print("specgram shape:",specgram.shape)
        p_wave = skimage.transform.rescale(wave, (specgram.shape[1]/wave.shape[0]))

        differences = []
        for value in p_wave:
            differences.append(abs(p_wave[0]-value))
        differences[0] = 10
        print(min(differences))
        index = differences.index(min(differences))
        print(p_wave[0])
        print("closest:",index)
        print(p_wave[index])
        print(p_wave[index-1])
        
        for n,value in enumerate(p_wave):
            if value == p_wave[0]:
                print(n)
        min_ = 10000
        for n,value in enumerate(differences):
            if value < min_:
                if p_wave[n-1] < p_wave[n]:
                    min_ = value
                    index = n
        print(min_)
        print("actual closest:",index)
        cut_spec = specgram[:index]
        print(cut_spec[-1])
        print(p_wave[473])
        print(p_wave[222])
        print(list(p_wave).index(np.min(np.abs(p_wave[:473:]))))
        print("pwave",p_wave.shape)
        specgram = np.real(np.transpose(cut_spec))
        print(np.median(specgram[0]))
        print(np.median(specgram[-1]))
        
        print(specgram.shape)
        #cont= input("...")
        specgram = specgram[:-1]
        print(specgram.shape)
        two = []
        for bin_ in specgram:
            two.append(bin_)
        for bin_ in specgram:
            two.append(bin_)
        specgram = np.stack(two)
        print(specgram.shape)
        t,back = istft(np.transpose(specgram),nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        sd.play(back,22050)
        cont = input("...")
        sd.play(back,22050)
        save_array(specgram, output_path+"/"+str(pitch))
        break
    
        

            

save_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/frequencies 2"
if not os.path.exists(save_path):
    os.makedirs(save_path)
generate_specgrams(save_path, 119)


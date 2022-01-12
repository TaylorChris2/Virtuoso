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

def midi_2_specgram(midi, freqs):
    specgram = np.zeros((freqs.shape[0], midi.shape[0]))
    for i,note in enumerate(midi):
        for channel in range(0,4):
            if note[0,channel] != 0:
                note_Hz = 440*(2**((note[0,channel]-69)/12)) ##scientific pitch to frequency formula
                for n, freq in enumerate(freqs):
                    if note_Hz < freq:
                        weight_freq = (note_Hz-freqs[n-1])/(freq-freqs[n-1])
                        weight_freq_n_1 = (freq-note_Hz)/(freq-freqs[n-1])
                        specgram[n][i] += weight_freq*note[1,channel]
                        specgram[n-1][i] += weight_freq_n_1*note[1,channel]
                        break
    return specgram
            


set_size = 48
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data"
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning"
frequency_clip = 1024 ##amount of frequencies to be included
time_split = 8192 ##milliseconds

midis = []
wavs = []
sets = 0
for set_ in os.listdir(path):
    print("\n"+set_+"\n")
    found_wav = False
    found_mid = False
    for file in os.listdir(os.path.join(path,set_)):
        if file.endswith(".wav") and not found_wav:
            y,sr = librosa.load(os.path.join(os.path.join(path,set_), file))
            Fs = 22050
            N = 2048
            w = np.hamming(N)
            ov = N - Fs // 1000
            f,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
            
            specgram = np.real(specgram)*0.1/np.max(np.real(specgram))
            print(np.min(specgram))
            specgram += 0.1
            specgram[specgram < 0] = 0
            wav_specgram = []
            for i in range(0,frequency_clip):
                wav_specgram.append(specgram[i])
            wav_specgram = np.stack(wav_specgram)
            found_wav = True
            print("Loaded wave file.")
            
        elif file.endswith("mid") and not found_mid:
            midi_array = load_midi_violin(os.path.join(os.path.join(path,set_), file))
            found_mid = True
            print("Loaded midi file.")

    if not found_wav or not found_mid:
        print("Data incomplete. Failed to load: " + os.path.join(path,set_))
    else:
        sets+=1

        specgram = midi_2_specgram(midi_array, f)
        specgram = skimage.transform.rescale(specgram, (1, (wav_specgram.shape[1]/specgram.shape[1])))
        midi_specgram = []
        for i in range(0,frequency_clip):
            midi_specgram.append(specgram[i])
        midi_specgram = np.stack(midi_specgram)
        midi_specgram *= 0.1
        midi_specgram += 0.1
        
        timef_midi = np.transpose(midi_specgram).astype("float16")
        timef_wav = np.transpose(wav_specgram).astype("float16")

        print("Converted to spectrogram.")

        
        index = 0
        segments = []
        while True:
            start = index*time_split
            end = (index+1)*time_split
            if np.array(timef_wav[start:end]).shape[0] == 0:
                break
            segments.append(np.array(timef_wav[start:end]))
            index += 1
        ##padding the ending
        padding_amt = time_split-segments[-1].shape[0]
        padding = np.zeros((padding_amt, segments[-1].shape[1]))
        new_last = []
        for time_ in segments[-1]:
            new_last.append(time_)
        for pad in padding:
            new_last.append(pad)
        segments[-1] = np.stack(new_last)
        for segment in segments:
            wavs.append(np.transpose(segment))

        print("Split wav spectrograms.")
        
        index = 0
        segments = []
        while True:
            start = index*time_split
            end = (index+1)*time_split
            if np.array(timef_midi[start:end]).shape[0] == 0:
                break
            segments.append(np.array(timef_midi[start:end]))
            index += 1
        ##padding the ending
        padding_amt = time_split-segments[-1].shape[0]
        padding = np.zeros((padding_amt, segments[-1].shape[1]))
        new_last = []
        for time_ in segments[-1]:
            new_last.append(time_)
        for pad in padding:
            new_last.append(pad)
        segments[-1] = np.stack(new_last)
        for segment in segments:
            midis.append(np.transpose(segment))

        print("Split midi spectrograms.")

        print("Loaded in" ,len(segments), "sets in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")

        ##playing the wavs for testing, not needed for data loading
        
        decoded = []
        for freq in wavs[0]:
            decoded.append(freq-0.1)
        decoded.append(np.zeros(time_split))
        decoded = np.stack(decoded)
        t,back = istft(np.abs(decoded),nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        sd.play(back,22050)
        
print("Loaded in" ,len(midis), "sets from", sets, "folders in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")
midi_sets, wav_sets = seperate_sets(midis, wavs, set_size)

start_time = time.time()


print("\nSaving loaded data in: " + save_folder_path + "...")

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

for n, set_ in enumerate(midi_sets):
    train_midis, val_midis, test_midis = split_train_val_test(set_)
    
    save_data_set(train_midis, save_folder_path, "Train Midis "+str(n))
    save_data_set(val_midis, save_folder_path, "Val Midis "+str(n))
    save_data_set(test_midis, save_folder_path, "Test Midis "+str(n))

print("Finished saving midis. Proceeding to save wavs...")

for n, set_ in enumerate(wav_sets):
    train_wavs, val_wavs, test_wavs = split_train_val_test(set_)
    
    save_data_set(train_wavs, save_folder_path, "Train Wavs "+str(n))
    save_data_set(val_wavs, save_folder_path, "Val Wavs "+str(n))
    save_data_set(test_wavs, save_folder_path, "Test Wavs "+str(n))

print("Finished saving wavs.")
print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
    int(((time.time() - start_time) % 60)+1), "seconds.")

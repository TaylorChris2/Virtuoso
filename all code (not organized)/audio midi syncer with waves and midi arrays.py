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
        track_array.append([[0.,0.,0.,0.],[0.,0.,0.,0.]])##one four channel list for pitch and one for articulation
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
                for slot in range(0,4):
                    if track_array[current_tick][1][slot] == 0:
                        track_array[current_tick][1][slot] = track[n+1].tick
                        break
                    
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
    wave = [i/((sample_rate/(2*np.pi))/freq) for i in range(0, int(duration))]
    wave = np.stack(wave)
    wave = np.cos(wave)
    '''
    sd.play(wave,sample_rate)
    cont = input("...")
    '''
    return wave

def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array


def save_array(array, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.exists(path):
        os.remove(path)
    hdf5_store = h5py.File(path, "a")
    hdf5_store.create_dataset("all_data", data = array, compression="gzip")
    hdf5_store.close()

def midi_2_note_number(midi, length_factor):
    midi_wave = np.zeros((4,int(midi.shape[0]*length_factor),2))
    last_print = 0
    for channel in range(0,4):
        note_num = 0
        current_tick = 0
        for i,note in enumerate(midi):
            if note[0,channel]>0: ## pitch
                try:
                    if note[1,channel] > 0: ## every note start
                        if i-last_print > 10000:
                            last_print = i
                        freq = 440*(2**((note[0,channel]-69)/12))              
                        wave = make_wave(freq, note[1,channel]*length_factor, 22050)
                        for j in range(0,len(wave)):
                            midi_wave[channel,current_tick+j,0]=freq
                            midi_wave[channel,current_tick+j,1]=note_num
                        current_tick += len(wave)
                        note_num += 1
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
    ## trimming nothingness
    first = len(midi_wave)-1
    last = 0
    found_first = False
    for n,note in enumerate(midi_wave[0]):
        if np.max(midi_wave[:,n,0]) > 0 and not found_first:
            found_first = True
            first = n
        elif found_first and np.max(midi_wave[:,n,0]) == 0:
            last = n
            break
    midi_wave = midi_wave[:,first:last,:]

    return midi_wave
            
def note_number_2_duration(note_number):
    durations = []
    last_print = 0
    for n,channel in enumerate(note_number):
        durations.append([])
        for i,note in enumerate(channel):
            if note_number[n,i-1,1] != note[1]: ##note start
                ind = 0
                duration = 1
                while True:
                    if note_number[n,i+ind,1] != note_number[n,(i+ind+1)%(note_number.shape[1]),1]:
                        break
                    ind += 1
                    duration += 1
                durations[n].append([note[0],i,duration])
    
    return durations

def rescale_duration_graph(duration, factor):
    rescaled = []
    for channel in duration:
        try:
            channel = np.stack(channel)
            channel[:,1:] *= factor
            rescaled.append(channel)
        except Exception as e:
            print(e)
            pass
    return rescaled
            
def duration_2_wave(duration, gradient_fraction = 3, return_different_gradients = True):
    midi_wave = []
    start_gradient = []
    end_gradient = []
    last = 0
    for n,channel in enumerate(duration):
        midi_wave.append(np.zeros(int(round(channel[-1,1]+channel[-1,2]))))
        start_gradient.append(np.ones(int(round(channel[-1,1]+channel[-1,2]))))
        end_gradient.append(np.ones(int(round(channel[-1,1]+channel[-1,2]))))
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if note[2] > 0: ## every note start
                        try:
                            duration = int(channel[i+1,1])-int(note[1])
                        except:
                            pass
                            duration = note[2]
                        wave = make_wave(note[0], duration, 22050)
                        gradient_amt = int(duration/gradient_fraction)
                        gradient = []
                        for g in range(0,gradient_amt):
                            gradient.append(g/gradient_amt)

                        for j,value in enumerate(wave):
                            midi_wave[n][int(note[1])+j]=wave[j]
                            if (int(note[1])+j) > last:
                                last = int(note[1])+j
                            try:
                                start_gradient[n][int(note[1])+j]=gradient[j]
                                end_gradient[n][int(channel[i+1,1])-j]=gradient[j]
                            except:
                                pass
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
                    
    midi_wave = midi_wave[:][:last+1]
    actual_wave = np.zeros(midi_wave[0].shape[0])
    for channel in midi_wave:
        actual_wave += channel
    if not return_different_gradients:
        return actual_wave
    else:
        start_gradient_wave = midi_wave.copy()
        actual_start_gradient_wave = np.zeros(midi_wave[0].shape[0])
        for n,channel in enumerate(start_gradient_wave):
            actual_start_gradient_wave += channel*start_gradient[n]
            
        end_gradient_wave = midi_wave.copy()
        actual_end_gradient_wave = np.zeros(midi_wave[0].shape[0])
        for n,channel in enumerate(end_gradient_wave):
            actual_end_gradient_wave += channel*end_gradient[n]
            
        end_start_gradient_wave = midi_wave.copy()
        actual_end_start_gradient_wave  = np.zeros(midi_wave[0].shape[0])
        for n,channel in enumerate(end_start_gradient_wave):
            actual_end_start_gradient_wave  += channel*start_gradient[n]*end_gradient[n]
        return actual_wave, actual_start_gradient_wave, actual_end_gradient_wave, actual_end_start_gradient_wave
               
sampling_size = 22050
path = path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 2"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000


if not os.path.exists(output+"/wavs"):
    os.makedirs(output+"/wavs")
if not os.path.exists(output+"/midis"):
    os.makedirs(output+"/midis")

midi_durations = []

sets = []

##change these if you want to continue
start_index = 0
start_midi_tick = 0
start_wav_tick = 0
save_index = 0
##


current_saved = 0
for set_ in os.listdir(path):
    sets.append(set_)
print(sets)

for set_num in range(start_index, len(sets)):
    print("\n"+sets[set_num]+"\n")
    found_wav = False
    found_mid = False
    for file in os.listdir(os.path.join(path,sets[set_num])):
        if file.endswith(".wav") and not found_wav:
            orig_wav,sr = librosa.load(os.path.join(os.path.join(path,sets[set_num]), file))
            wav_length = orig_wav.shape[0]
            found_wav = True
            print("Loaded wave file.")
            
        elif file.endswith("mid") and not found_mid:
            midi_array = load_midi_violin(os.path.join(os.path.join(path,sets[set_num]), file))
            print(midi_array.shape)
            found_mid = True
            print("Loaded midi file.")
    length_factor = (wav_length/midi_array.shape[0])
    orig_midi_note_number = midi_2_note_number(midi_array, length_factor)
    for channel in orig_midi_note_number:
        for thing in channel:
            pass
            #print(thing)
    wav_tick = start_wav_tick
    midi_tick = start_midi_tick
    start_midi_tick = 0
    start_wav_tick = 0
    while True:
        print("0")
        good = ""
        actual_midi_note_num = orig_midi_note_number[:,midi_tick:]
        print("1")
        actual_wav = orig_wav[wav_tick:]
        print("2")
        actual_midi_duration = note_number_2_duration(actual_midi_note_num)
        print(len(actual_midi_duration),len(actual_wav),len(actual_midi_note_num))
        print("midi tick:",midi_tick,"wav tick:",wav_tick)
        if actual_wav.shape[0] == 0:
            break
        good = float(input("enter a factor:"))
        while good != "y":
            real_length = round(sampling_size/float(good))
            actual_factor = float(good)
            print(real_length)
            midi_duration = rescale_duration_graph(actual_midi_duration, (real_length/sampling_size))
            print(len(midi_duration[0]))
            midi,midi_start,midi_end,midi_start_end  = duration_2_wave(midi_duration)
            midi = midi*0.1/np.max(midi)
            midi_start = midi_start*0.1/np.max(midi_start)
            midi_end = midi_end*0.1/np.max(midi_end)
            midi_start_end = midi_start_end*0.1/np.max(midi_start_end)
            if midi.shape[0] > sampling_size:
                midi = midi[:sampling_size]
            elif midi.shape[0] < sampling_size:
                padding_amt = sampling_size-midi.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in midi:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi = np.stack(padded)

            if midi_start.shape[0] > sampling_size:
                midi_start = midi_start[:sampling_size]
            elif midi_start.shape[0] < sampling_size:
                padding_amt = sampling_size-midi_start.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in midi_start:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi_start = np.stack(padded)

            if midi_start_end.shape[0] > sampling_size:
                midi_start_end = midi_start_end[:sampling_size]
            elif midi_start_end.shape[0] < sampling_size:
                padding_amt = sampling_size-midi_start_end.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in midi_start_end:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi_start_end = np.stack(padded)

            ## for the flattened version    
            if midi_end.shape[0] > sampling_size:
                midi_end = midi_end[:sampling_size]
            elif midi_end.shape[0] < sampling_size:
                padding_amt = sampling_size-midi_end.shape[0]
                padding = np.zeros((padding_amt))
                padded = []
                for time_ in midi_end:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi_end = np.stack(padded)

            wav = actual_wav[:sampling_size]
            midi = midi[:sampling_size]
            midi_end = midi_end[:sampling_size]
            midi_start_end = midi_start_end[:sampling_size]
            midi_start = midi_start[:sampling_size]

            added = wav+midi_end
            sd.play(added,5512)
            good = input("did it sound good? (enter next factor for no) (yes for good) (b to go back one) (r to replay):")
            while good == "r":
                speed = int(input("what speed do you wanted it played at?:"))
                sd.play(added,int(22050/speed))
                good = input("did it sound good? (enter next factor for no) (yes for good) (b to go back one) (r to replay):")
            if good == "b":
                backing = True
                break
            else:
                backing = False
            print(midi.shape)
        print("exited")
        if not backing:
            save_array(wav, output+"/wavs/"+str(save_index)+".h5")
            save_array(midi, output+"/midis/no gradient/"+str(save_index)+".h5")
            save_array(midi_start, output+"/midis/start gradient/"+str(save_index)+".h5")
            save_array(midi_end, output+"/midis/end gradient/"+str(save_index)+".h5")
            save_index+=1
            wav_tick += sampling_size
            midi_durations.append(real_length)
            midi_tick += real_length
        else:
            save_index-=1
            wav_tick -= sampling_size
            midi_tick -= midi_durations[-1]
            del midi_durations[-1]


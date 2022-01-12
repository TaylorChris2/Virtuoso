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
    for slot in range(0,4):
        current_tick = 0
        for n,event in enumerate(only_notes[slot]):
            current_tick += event.tick
            if event.data[1] == 100:##every note start
                track_array[current_tick][1][slot] = only_notes[slot][n+1].tick
                for i in range(current_tick,current_tick+only_notes[slot][n+1].tick):
                    track_array[i][0][slot] = event.data[0]
                    
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

def midi_2_note_number(midi, length_factor, gradient_fraction = 3):
    midi_wave = np.zeros((4,int(midi.shape[0]*length_factor*1.5),2))
    start_gradients = np.ones((4,int(midi.shape[0]*length_factor*1.5)))
    end_gradients = np.ones((4,int(midi.shape[0]*length_factor*1.5)))
    last_print = 0
    for channel in range(0,4):
        note_num = 1
        current_tick = 0
        first_note = False
        for i,note in enumerate(midi):
            if note[0,channel]>0: ## pitch
                try:
                    if note[1,channel] > 0: ## every note start
                        current_tick = int(round(i*length_factor))
                        if i-last_print > 10000:
                            last_print = i
                        freq = 440*(2**((note[0,channel]-69)/12))
                        general_gradient_amt = int(round(round(note[1,channel]*length_factor)/gradient_fraction))
                        general_gradient = []
                        for g in range(0,general_gradient_amt):
                            general_gradient.append(g/general_gradient_amt)
                        for j in range(0,int(round(note[1,channel]*length_factor))):
                            midi_wave[channel,current_tick+j,0]=freq
                            midi_wave[channel,current_tick+j,1]=note_num
                            try:
                                start_gradients[channel,current_tick+j]=general_gradient[j]
                                end_gradients[channel,current_tick+int(round(note[1,channel]*length_factor))-1-j]=general_gradient[j]
                            except Exception as e:
                                #print(e)
                                pass
                        note_num += 1
                        
                except Exception as e:
                    print(e)
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
    start_gradients = start_gradients[:,first:last]
    end_gradients = end_gradients[:,first:last]

    return midi_wave, start_gradients, end_gradients

            
def note_number_2_duration(note_number):
    durations = []
    last_print = 0
    print(note_number[0,0],note_number[0,-1])
    for n,channel in enumerate(note_number):
        durations.append([])
        for i,note in enumerate(channel):
            if note_number[n,i-1,1] != note[1] and note[1] != 0: ##note start
                ind = 0
                duration = 1
                while True:
                    try:
                        if note_number[n,i+ind,1] != note_number[n,(i+ind+1)%(note_number.shape[1]),1]:
                            break
                        ind += 1
                        duration += 1
                    except Exception as e:
                        print(e)
                        print(note_number[n,i+ind-1,1])
                        print(note_number[n,(i+ind)%(note_number.shape[1]),1])
                        print(i+ind-1)
                        print((i+ind)%(note_number.shape[1]))
                        print("")

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

def duration_2_note_number(duration, gradient_fraction = 3):
    midi_wave = []
    start_gradients = []
    end_gradients = []
    last = 0
    lengths = []
    for n,channel in enumerate(duration):
        lengths.append(int(round(channel[-1][1]+channel[-1][2])))
    length = np.max(lengths)
    for n,channel in enumerate(duration):
        note_num = 0
        midi_wave.append(np.zeros((length, 2)))
        start_gradients.append(np.ones(length))
        end_gradients.append(np.ones(length))
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if note[2] > 0: ## every note start
                        try:
                            wave_duration = int(channel[i+1][1])-int(note[1])
                        except:
                            pass
                            wave_duration = note[2]
                        general_gradient_amt = int(wave_duration/gradient_fraction)
                        general_gradient = []
                        for g in range(0,general_gradient_amt):
                            general_gradient.append(g/general_gradient_amt)
                        for j in range(0,int(wave_duration)):
                            midi_wave[n][int(note[1])+j,0]=note[0]
                            midi_wave[n][int(note[1])+j,1]=note_num
                            if (int(note[1])+j) > last:
                                last = int(note[1])+j
                            try:
                                start_gradients[n][int(note[1])+j]=general_gradient[j]
                                end_gradients[n][int(note[1])+wave_duration-1-j]=general_gradient[j]
                            except:
                                pass
                        note_num += 1
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
    return np.stack(midi_wave[:][:last+1]), np.stack(start_gradients[:][:last+1]), np.stack(end_gradients[:][:last+1])
            
def duration_2_wave(duration, gradient_fraction = 3, return_different_gradients = False, gradients = None):
    midi_wave = []
    last = 0
    lengths = []
    for n,channel in enumerate(duration):
        lengths.append(int(round(channel[-1][1]+channel[-1][2])))
    length = np.max(lengths)
    for n,channel in enumerate(duration):
        midi_wave.append(np.zeros(length))
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if note[2] > 0: ## every note start
                        try:
                            wave_duration = int(channel[i+1][1])-int(note[1])
                        except:
                            print("eff off",note[1])
                            pass
                            wave_duration = note[2]
                        wave = make_wave(note[0], wave_duration, 22050)
                        for j,value in enumerate(wave):
                            midi_wave[n][int(note[1])+j]=value
                            if (int(note[1])+j) > last:
                                last = int(note[1])+j
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
                    
    midi_wave = midi_wave[:][:last+1]
    actual_wave = np.zeros(midi_wave[0].shape[0])
    for n,channel in enumerate(midi_wave):
        for j,value in enumerate(channel):
            if value != 0:
                print("first thing:", value, j)
                break
        print(channel[:50])
        if gradients is not None:
            print("using gradients")
            for gradient in gradients:
                channel*=gradient[n]
        actual_wave += channel
    return actual_wave
               
sampling_size = 882000
path = path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 2"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000


midi_durations = []

sets = []

##change these if you want to continue
start_index = 0
start_midi_tick = 0
start_wav_tick = 8000
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
            for note in midi_array:
                pass
                #print(note)
            print(midi_array.shape)
            found_mid = True
            print("Loaded midi file.")
    length_factor = int(wav_length/midi_array.shape[0])
    print("\n\nlength factor:",length_factor,"\n\n")
    orig_midi_note_number, orig_start_gradients, orig_end_gradients = midi_2_note_number(midi_array, length_factor)
    for n,note in enumerate(orig_midi_note_number[0]):
        pass
    #print(orig_midi_note_number[:,n])
    orig_midi_duration = note_number_2_duration(orig_midi_note_number)
    for channel in orig_midi_duration:
        print("\n", channel)
    wave = duration_2_wave(orig_midi_duration, gradients = [orig_start_gradients,orig_end_gradients])
    wave = 0.1*wave/np.max(wave)
    cont = ""
    while cont != "n:":
        sd.play(wave,22050)
        cont = input("...")
    print(orig_midi_duration[1],"\n",orig_midi_duration[2],"\n",orig_midi_duration[3])
    for channel in orig_midi_note_number:
        for n,thing in enumerate(channel):
            pass
            #print(orig_midi_note_number[:,n,:])
    
    wav_tick = start_wav_tick
    midi_tick = start_midi_tick
    start_midi_tick = 0
    start_wav_tick = 0
    while True:
        print("midi tick:",midi_tick,"wav tick:",wav_tick)
        good = float(input("enter a factor:"))
        while good != "y":
            real_length = round(sampling_size/float(good))
            actual_midi_tick = int(round(midi_tick*float(good)))
            print(real_length)
            print(actual_midi_tick)
            midi_duration = rescale_duration_graph(orig_midi_duration, (sampling_size/real_length))
            print(len(midi_duration))
            midi_graph, start_gradient_graph, end_gradient_graph = duration_2_note_number(midi_duration)
            print("midi graph shape:",midi_graph.shape)
            midi_wave  = duration_2_wave(midi_duration, gradients = [end_gradient_graph])
            
            midi_graph = midi_graph[:,actual_midi_tick:actual_midi_tick+sampling_size]
            print("midi graph shape:",midi_graph.shape)
            start_gradient_graph = start_gradient_graph[:,actual_midi_tick:actual_midi_tick+sampling_size]
            print("start gradient graph shape:",start_gradient_graph.shape)
            end_gradient_graph = end_gradient_graph[:,actual_midi_tick:actual_midi_tick+sampling_size]
            print("end gradient graph shape:",end_gradient_graph.shape)
            
            midi_wave = midi_wave[actual_midi_tick:actual_midi_tick+sampling_size]
            midi_wave = midi_wave*0.1/np.max(midi_wave)
            if midi_wave.shape[0] < sampling_size:
                padding_amt = sampling_size-midi.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in midi_wave:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi_wave = np.stack(padded)
            print(midi_wave.shape)

            wav = orig_wav[wav_tick:wav_tick+sampling_size]

            added = midi_wave+wav
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
        print("exited")
        if not backing:
            save_array(wav, output+"/wavs/"+str(save_index)+".h5")
            save_array(midi_graph, output+"/midis/no gradient/"+str(save_index)+".h5")
            save_array(start_gradient_graph, output+"/midis/start gradient graphs/"+str(save_index)+".h5")
            save_array(end_gradient_graph, output+"/midis/end gradient graphs/"+str(save_index)+".h5")
            save_index+=1
            wav_tick += sampling_size
            midi_durations.append(real_length)
            midi_tick += real_length
        else:
            save_index-=1
            wav_tick -= sampling_size
            midi_tick -= midi_durations[-1]
            del midi_durations[-1]


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
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import h5py
import scipy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from scipy.signal import istft
from scipy.signal import stft
import random

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
                            track_array[current_tick][1][slot] = track[n+1].tick
                            break
                    
    return track_array     


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
                        if not first_note:
                            current_tick = i
                            first_note = True
                        if i-last_print > 10000:
                            last_print = i
                        freq = 440*(2**((note[0,channel]-69)/12))
                        general_gradient_amt = int(int(note[1,channel]*length_factor)/gradient_fraction)
                        general_gradient = []
                        for g in range(0,general_gradient_amt):
                            general_gradient.append(g/general_gradient_amt)
                        for j in range(0,int(note[1,channel]*length_factor)):
                            midi_wave[channel,current_tick+j,0]=freq
                            midi_wave[channel,current_tick+j,1]=note_num
                            try:
                                start_gradients[channel,current_tick+j]=general_gradient[j]
                                end_gradients[channel,current_tick+int(note[1,channel]*length_factor)-1-j]=general_gradient[j]
                            except Exception as e:
                                #print(e)
                                pass
                        current_tick += int(note[1,channel]*length_factor)
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
    stacked = []
    for channel in durations:
        try:
            channel = np.stack(channel)
            stacked.append(channel)
        except Exception as e:
            print(e)
            pass
    return stacked

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
        lengths.append(int(round(channel[-1,1]+channel[-1,2])))
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
                            wave_duration = int(channel[i+1,1])-int(note[1])
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
        lengths.append(int(round(channel[-1,1]+channel[-1,2])))
    length = np.max(lengths)
    for n,channel in enumerate(duration):
        midi_wave.append(np.zeros(length))
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if note[2] > 0: ## every note start
                        try:
                            wave_duration = int(channel[i+1,1])-int(note[1])
                        except:
                            print("eff off",note[1])
                            pass
                            wave_duration = note[2]
                        wave = make_wave(note[0], wave_duration*1.1, 22050)
                        for j,value in enumerate(wave):
                            try:
                                midi_wave[n][int(note[1])+j]=value
                                if (int(note[1])+j) > last:
                                    last = int(note[1])+j
                            except:
                                pass
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
                    
    midi_wave = midi_wave[:][:last+1]
    actual_wave = np.zeros(midi_wave[0].shape[0])
    for n,channel in enumerate(midi_wave):
        print(channel[:50])
        if gradients is not None:
            for gradient in gradients:
                channel*=gradient[n]
        actual_wave += channel
    return actual_wave

def make_wave(freq, duration, sample_rate = 22050):
    wave = [i/((sample_rate/(2*np.pi))/freq) for i in range(0, int(duration))]
    wave = np.stack(wave)
    wave = np.cos(wave)
    '''
    sd.play(wave,sample_rate)
    cont = input("...")
    '''
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

def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array

def predict_midi(midi, continuous_model):
    print("\n\n\n\n shape",np.stack([np.stack([midi],axis=2)]).shape)
    continuous_graph = continuous_model.predict(np.stack([np.stack([midi],axis=2)]))
    continuous_array = np.squeeze(continuous_graph[0], axis=2)

    actual_array = continuous_array.copy()
    #actual_array[actual_array < 0] = 0
    print("min and max of actual array:",np.min(actual_array),np.max(actual_array))
    #actual_array += np.min(actual_array)
    #actual_array[binary_array == 0] = 0
    '''fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(midi)
    plt.show()'''

    specgram = np.transpose(actual_array)
    decoded = []
    for freq in specgram:
        decoded.append(freq)
    for i in range(0,(1025-frequency_clip_wav)):
        decoded.append(np.zeros(specgram.shape[1]))
    decoded = np.stack(decoded)
    decoded = (decoded*100)-100
    decoded = librosa.db_to_amplitude(decoded)

    return decoded

           

def load_midi(path):
    frequency_clip_midi = 512
    midi_array = load_midi_violin(path)
    length_factor = 20
    orig_midi_note_number, orig_start_gradients, orig_end_gradients = midi_2_note_number(midi_array, length_factor)
    orig_midi_duration = note_number_2_duration(orig_midi_note_number)
    midi_duration = rescale_duration_graph(orig_midi_duration, 1)
    midi_graph, start_gradient_graph, end_gradient_graph = duration_2_note_number(midi_duration)

    midi_duration = note_number_2_duration(np.stack(midi_graph))

    midi_wave = duration_2_wave(midi_duration, gradients = [start_gradient_graph,end_gradient_graph])
    midi_wave = midi_wave*0.1/np.max(midi_wave)
    #sd.play(midi_wave,22050)
    #cont = input("go?:")

    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    f,t,specgram = stft(midi_wave,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    specgram = np.real(specgram)
    specgram[specgram < 0] = 0
    specgram = librosa.amplitude_to_db(specgram, top_db=None)


    midi_specgram = []
    for i in range(0,frequency_clip_midi):
        midi_specgram.append(specgram[i])
    midi_specgram = np.stack(midi_specgram)

    midi_specgram += 100
    midi_specgram = midi_specgram/100


    timef_midi = np.transpose(midi_specgram)

    return timef_midi, midi_wave

slide_window = 512

def down_block(x, filters, dropout, kernel_size=(3,3), padding="same", strides=1, pool_size = (2,2)):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    p = keras.layers.MaxPool2D(pool_size, pool_size)(c)
    return c, p

def up_block(x, skip, filters, dropout, kernel_size=(3,3), padding="same", strides=1, pool_size = (2,2)):
    up = keras.layers.UpSampling2D(pool_size)(x)
    concat = keras.layers.Concatenate()([up, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = concat.shape[1:], kernel_initializer='he_normal')(concat)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def bottleneck(x, filters, dropout, kernel_size=(3,3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = x.shape[1:], kernel_initializer='he_normal')(x)
    c = keras.layers.Dropout(dropout)(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu", input_shape = c.shape[1:], kernel_initializer='he_normal')(c)
    return c

def ConvNetContinuous(x,y):
    inputs = keras.layers.Input((x, y, 1))
    
    #conving input
    p0 = inputs
    c1, p1 = down_block(p0, 16, 0.1)
    print(p1.shape)
    c2, p2 = down_block(p1, 32, 0.1) 
    print(p2.shape)
    c3, p3 = down_block(p2, 64, 0.2)
    print(p3.shape)
    c4, p4 = down_block(p3, 128, 0.2)
    c5, p5 = down_block(p4, 256, 0.3)
    c6, p6 = down_block(p5, 512, 0.4)
    print(p4.shape)
    #bottleneck (im not completely sure what this does but apparently it's important and it sucks w/o it so)
    bn = bottleneck(p6, 1024, 0.5)
    print(bn.shape)
    #up-conving for output
    u1 = up_block(bn, c6, 512, 0.4)
    u2 = up_block(u1, c5, 256, 0.4)
    u3 = up_block(u2, c4, 128, 0.3)
    print(u1.shape)
    u4 = up_block(u3, c3, 64, 0.2)
    print(u2.shape)
    u5 = up_block(u4, c2, 32, 0.2) 
    print(u3.shape)
    u6 = up_block(u5, c1, 16, 0.1)
    print(u4.shape)

    outputs = keras.layers.Conv2D(1, (1,1), padding="same")(u6)
    print("out:",outputs.shape)

    model = keras.models.Model(inputs, outputs)
    return model

def load_wave(path):
    complete_wave = []
    file = 1
    while True:
        try:
            wave_array = load_array(path+"/"+str(file)+".h5")    
            for moment in wave_array:
                complete_wave.append(moment)
            file+=1
        except:
            break
    complete_wave = np.stack(complete_wave)
    return complete_wave

def load_graph(path):
    complete_graph = []
    for i in range(0, load_array(path+"/"+os.listdir(path)[0]).shape[0]):
        complete_graph.append([])
    file = 1
    while True:
        try:
            array = load_array(path+"/"+str(file)+".h5")
            for n,channel in enumerate(array):
                for moment in channel:
                    complete_graph[n].append(moment)
            file+=1
        except:
            break
    complete_graph = np.stack(complete_graph)
    return complete_graph

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


##############################      pathes to change      ################################  
continuous_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Saved Models/continuous both gradient 512 1/Model 51 (2).h5"
continuous_model = ConvNetContinuous(512, 512)
continuous_model.load_weights(continuous_path)
save_folder = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/test prediction waves"
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/0"
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning continuous both gradient"

print("loaded models")

input_midi_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 2/S3 Alg 1 of 4/allegro assai 1 of 4.mid"
##############################      pathes to change      ################################  

frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included

set_size = 512
sets = 0

Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000

time_split = 512

frequency_clip_wav = 512
slide_window = int(time_split/2)



y = load_wave(path+"/wavs")
y = y*0.1/np.max(y)
wav_length = y.shape[0]
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
specgram = np.real(specgram)
specgram[specgram < 0] = 0
specgram = librosa.amplitude_to_db(specgram, top_db=None)


wav_specgram = []
for i in range(0,frequency_clip_wav):
    wav_specgram.append(specgram[i])
wav_specgram = np.stack(wav_specgram)

wav_specgram += 100
wav_specgram = wav_specgram/100

print(wav_specgram.shape)

#wav_specgram = 10**wav_specgram
print(np.max(wav_specgram))
print(np.min(wav_specgram))
'''
extent = [0,8192,0,1024]
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111)
im = ax.imshow(wav_specgram, extent=extent, origin='lower')
plt.show()
'''
print("Loaded wave file.")

###loading in midi------------------------------------------------------------
midi_graph = load_graph(path+"/midis/no gradient")
start_gradient_graph = load_graph(path+"/midis/start gradient graphs")
end_gradient_graph = load_graph(path+"/midis/end gradient graphs")

midi_duration = note_number_2_duration(midi_graph)

midi_wave = duration_2_wave(midi_duration, gradients = [start_gradient_graph, end_gradient_graph])
midi_wave = midi_wave*0.1/np.max(midi_wave)
#sd.play(midi_wave,22050)
#cont = input("...")

Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f,t,specgram = stft(midi_wave,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
specgram = np.real(specgram)
specgram[specgram < 0] = 0
specgram = librosa.amplitude_to_db(specgram, top_db=None)


midi_specgram = []
for i in range(0,frequency_clip_midi):
    midi_specgram.append(specgram[i])
midi_specgram = np.stack(midi_specgram)

midi_specgram += 100
midi = midi_specgram/100

midi = np.transpose(midi)

print("\n\nBEFORE:",midi.shape,"\n\n")


slide_window = 512
index = 0
midi_segments = []
while True:
    print(index/len(midi))
    segment = midi[index:index+time_split]
    if segment.shape[0] == 0:
        break
    if segment.shape[0] < time_split:
        padded = []
        for moment in segment:
            padded.append(moment)
        pad_shape = [segment.shape]
        pad_shape[0] = 1
        for i in range(0,time_split-segment.shape[0]):
            padded.append(np.zeros(segment.shape[1]))
        segment = np.stack(padded)
    print(segment.shape)
    continuous_graph = continuous_model.predict(np.stack([np.stack([segment],axis=2)]))
    specgram = np.transpose(np.squeeze(continuous_graph[0],axis=2))

    decoded = []
    for freq in specgram:
        decoded.append(freq)
    for i in range(0,(1025-frequency_clip_wav)):
        decoded.append(np.zeros(specgram.shape[1]))
    decoded = np.stack(decoded)
    decoded = (decoded*100)-100
    decibels = librosa.db_to_amplitude(decoded)
        
    midi_segments.append(decibels)
    index += slide_window

complete_specgram = []
for n,segment in enumerate(midi_segments):
    print(n/len(midi_segments))
    segment = np.transpose(segment)
    for moment in segment:
        complete_specgram.append(moment)

complete_specgram = np.transpose(np.stack(complete_specgram))
print("\n\ncomplete specgram shape:",complete_specgram.shape,"\n\n")
    
t,back = istft(complete_specgram,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
back = back*0.1/np.max(back)
back[abs(back) < 0.1] == 0

Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f,t,specgram = stft(back,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
specgram = np.real(specgram)
specgram[specgram < 0] = 0
specgram = librosa.amplitude_to_db(specgram, top_db=None)


midi_specgram = []
for i in range(0,frequency_clip_midi):
    midi_specgram.append(specgram[i])
midi_specgram = np.stack(midi_specgram)

midi_specgram += 100
midi_specgram = midi_specgram/100

print("Loaded midi file.")


if np.min(midi_specgram) < 0 or np.min(wav_specgram) < 0:
    print("\n\nNOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n\n")
       
timef_midi = np.transpose(midi_specgram)

print("\n\nAFTER:",timef_midi.shape,"\n\n")


#sd.play(back,22050)
#cont = input("bob: ")


print("Loaded midi file.")


if np.min(midi_specgram) < 0 or np.min(wav_specgram) < 0:
    print("\n\nNOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n\n")
       

timef_wav = np.transpose(wav_specgram)


print("specgram shapes:", timef_midi.shape,timef_wav.shape)
print(np.max(timef_wav))
print(np.min(timef_wav))

print("Converted to spectrogram.")
delete_last = False


print("Split wav spectrograms.")

time_split = 2048
midis = []
wavs = []

index = 0
segments = []
start = 0
end = time_split
while True:
    segments.append(np.array(timef_midi[start:end]))
    start += slide_window
    end += slide_window
    if np.array(timef_midi[start:end]).shape[0] < time_split:
        break
##padding the ending
'''if segments[-1].shape[0] > 1000:
    padding_amt = time_split-segments[-1].shape[0]
    padding = np.zeros((padding_amt, segments[-1].shape[1]))
    new_last = []
    for time_ in segments[-1]:
        new_last.append(time_)
    for pad in padding:
        #print("pad",pad)
        new_last.append(pad)
    segments[-1] = np.stack(new_last)
else:
    print(segments[-1].shape)
    del segments[-1]
    delete_last = True      '''
del segments[-1]
for segment in segments:
    midis.append(segment)



    
index = 0
segments = []
start = 0
end = time_split
while True:
    segments.append(np.array(timef_wav[start:end]))
    start += slide_window
    end += slide_window
    if np.array(timef_wav[start:end]).shape[0] < time_split:
        break
if not delete_last:
    padding_amt = time_split-segments[-1].shape[0]
    padding = np.zeros((padding_amt, segments[-1].shape[1]))
    new_last = []
    for time_ in segments[-1]:
        new_last.append(time_)
    for pad in padding:
        new_last.append(pad)
    segments[-1] = np.stack(new_last)
else:
    print("DELETING LAST, LESS THAN 3 SECONDS LONG")
    del segments[-1]
    delete_last = True
for segment in segments:
    wavs.append(segment)
fig = plt.figure()
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(midis[-1])
ax = fig.add_subplot(1, 1, 1)
ax.imshow(wavs[-1])
plt.show()

print("lengths: ",len(midis),len(wavs))
print("Split midi spectrograms.")

print("Loaded in" ,len(segments), "sets in", int((time.time() - start_time)/60), "minutes and",
  int(((time.time() - start_time) % 60)+1), "seconds.")

'''
for n, wav in enumerate(wavs):
    print(wav.shape)
    print(wav.dtype)
    fig = plt.figure()
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(wav)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(midis[n])
    plt.show()
'''
##playing the wavs for testing, not needed for data loading

decoded = []
converted_back_midi = np.transpose(timef_wav)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(timef_wav)
plt.show()

#converted_back_midi[converted_back_midi < 0.3] = 0
print("converted unique:",np.unique(converted_back_midi))
decoded = []
for freq in converted_back_midi:
    decoded.append(freq)
for i in range(0,(1025-frequency_clip_midi)):
    decoded.append(np.zeros(converted_back_midi.shape[1]))
decoded = np.stack(decoded)
decoded = (decoded*100)-100
decoded = librosa.db_to_amplitude(decoded)
print(decoded.shape)
t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
back1 = back*0.1/np.max(back)
print(back[-1])
sd.play(back1,22050)
#time.sleep(5)
#cont = input("...;")
'''
converted_back_wav = np.transpose(timef_wav)*2
converted_back_wav = (converted_back_wav-1) + converted_back_midi  XD
'''
converted_back_wav = np.transpose(timef_wav)
print("converted shape:",converted_back_wav.shape)
decoded = []
for freq in converted_back_wav:
    decoded.append(freq)
for i in range(0,(1025-frequency_clip_wav)):
    decoded.append(np.zeros(converted_back_wav.shape[1]))
decoded = np.stack(decoded)
decoded = (decoded*100)-100
decoded = librosa.db_to_amplitude(decoded)
t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
back = back*0.1/np.max(back)
print(back[-1])
sd.play(back,22050)
#cont = input("...")

'''while cont != "n":
    sd.play(back1,22050)
    cont = input("...")
    sd.play(back,22050)
    cont = input("...")'''
    
    
new_indexes = []
for i in range(0,len(midis)):
    index = random.randint(0,len(midis)-1)
    while index in new_indexes:
        index = random.randint(0,len(midis)-1)
    new_indexes.append(index)

print(new_indexes)
print(len(midis))

new_midis = []
new_wavs = []
for index in new_indexes:
    print(index)
    new_midis.append(midis[index])
    new_wavs.append(wavs[index])

    
        
print("Loaded in" ,len(midis),len(wavs), "sets from", sets, "folders in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")
midi_sets, wav_sets = seperate_sets(new_midis, new_wavs, set_size)
print(len(midi_sets))

start_time = time.time()


print("\nSaving loaded data in: " + save_folder_path + "...")

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

for n, set_ in enumerate(midi_sets):
    train_midis, val_midis, test_midis = split_train_val_test(set_)
    print(len(train_midis), len(val_midis), len(test_midis))
    
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



scipy.io.wavfile.write(save_folder+"/prediction_7.wav",22050,np.array(back*32767, dtype=np.int16))
scipy.io.wavfile.write(save_folder+"/input.wav",22050,np.array(orig_midi*32767, dtype=np.int16))


while True:
    sd.play(back, 22050)
    cont = input(":")

for segment in midi_segments:
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(segment)
    plt.show()


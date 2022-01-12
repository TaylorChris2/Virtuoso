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
import tensorflow.keras as keras
start_time = time.time()


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
    while True:
        try:
            if os.path.exists(path):
                os.remove(path)

            hdf5_store = h5py.File(path, "a")
            hdf5_store.create_dataset("all_data", data = array, compression="gzip")
            break
        except:
            pass

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
                            duration = int(channel[i+1,1])-int(note[1])
                        except:
                            pass
                            duration = note[2]
                        wave = make_wave(note[0], duration, 22050)
                        for j,value in enumerate(wave):
                            midi_wave[n][int(note[1])+j]=wave[j]
                            if (int(note[1])+j) > last:
                                last = int(note[1])+j
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
                    
    midi_wave = midi_wave[:][:last+1]
    actual_wave = np.zeros(midi_wave[0].shape[0])
    for n,channel in enumerate(midi_wave):
        if gradients is not None:
            for gradient in gradients:
                channel*=gradient[n]
        actual_wave += channel
    return actual_wave


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

def ConvNet(x,y):
    inputs = keras.layers.Input((x, y, 1))
    
    #conving input
    p0 = inputs
    c1, p1 = down_block(p0, 8, 0.1)
    print(p1.shape)
    c2, p2 = down_block(p1, 16, 0.1) 
    print(p2.shape)
    c3, p3 = down_block(p2, 32, 0.2)
    print(p3.shape)
    c4, p4 = down_block(p3, 64, 0.2)
    c5, p5 = down_block(p4, 128, 0.3)
    print(p4.shape)
    #bottleneck (im not completely sure what this does but apparently it's important and it sucks w/o it so)
    bn = bottleneck(p5, 256, 0.4)
    print(bn.shape)
    #up-conving for output
    u1 = up_block(bn, c5, 128, 0.3)
    u2 = up_block(u1, c4, 64, 0.2)
    print(u1.shape)
    u3 = up_block(u2, c3, 32, 0.2)
    print(u2.shape)
    u4 = up_block(u3, c2, 16, 0.1) 
    print(u3.shape)
    u5 = up_block(u4, c1, 8, 0.1)
    print(u4.shape)

    outputs = keras.layers.Conv2D(1, (1,1), padding="same")(u5)
    print("out:",outputs.shape)

    model = keras.models.Model(inputs, outputs)
    return model

slide_window = 512


set_size = 128
continuous_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Saved Models/continuous both gradient 1/Model 81 (2).h5"
continuous_model = ConvNet(2048, 512)
continuous_model.load_weights(continuous_path)
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/0"
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning continuous both gradient"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
time_split = 2048 ##milliseconds

midis = []
wavs = []
sets = 0
sets_ = []
start_index = 0
for set_ in os.listdir(path):
    sets_.append(set_)
print(sets_)
for set_num in range(0,1):
    ###loading in spectrograms-----------------------------------------------------------
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

    #midi_duration = rescale_duration_graph(midi_duration, 1.2)

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
    midi_specgram = midi_specgram/100
    
    print("Loaded midi file.")


    sets+=1
    if np.min(midi_specgram) < 0 or np.min(wav_specgram) < 0:
        print("\n\nNOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n\n")
           
    timef_midi = np.transpose(midi_specgram)
    timef_wav = np.transpose(wav_specgram)

 
    print("specgram shapes:", timef_midi.shape,timef_wav.shape)
    print(np.max(timef_wav))
    print(np.min(timef_wav))

    print("Converted to spectrogram.")
    delete_last = False


    print("Split wav spectrograms.")
    
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
    if segments[-1].shape[0] > 1000:
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
        delete_last = True            
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
    print("Split midi spectrograms.")

    print("Loaded in" ,len(segments), "sets in", int((time.time() - start_time)/60), "minutes and",
      int(((time.time() - start_time) % 60)+1), "seconds.")

    actual_midis = []
    ##playing the wavs for testing, not needed for data loading
    print(np.stack([np.stack([midis[0]],axis=2)]))
    for n,midi in enumerate(midis):
        continuous_graph = continuous_model.predict(np.stack([np.stack([midi],axis=2)]))
        continuous_array = np.squeeze(continuous_graph[0], axis=2)
        actual_midis.append(continuous_array)

    

    '''decoded = []
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

    ''''''
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
    cont = input("...")

    while cont != "n":
        sd.play(back1,22050)
        cont = input("...")
        sd.play(back,22050)
        cont = input("...")'''
        
        
        
    
        
print("Loaded in" ,len(actual_midis),len(wavs), "sets from", sets, "folders in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")
midi_sets, wav_sets = seperate_sets(actual_midis, wavs, set_size)

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
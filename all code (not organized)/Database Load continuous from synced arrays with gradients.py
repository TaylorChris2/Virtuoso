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
import random
start_time = time.time()

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
                    try:
                        if note_number[n,i+ind,1] != note_number[n,(i+ind+1)%(note_number.shape[1]),1]:
                            break
                        ind += 1
                        duration += 1
                    except:
                        break
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
    file = 0
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
    file = 0
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

def note_number_to_wave(note_number, gradient_fraction=3, end_gradient = True, start_gradient = True, rescale_factor=1):
    last = 0
    rescaled_note_number = np.round(skimage.transform.rescale(note_number, (1, rescale_factor, 1)))
    midi_wave = rescaled_note_number.copy()[:,:,0]
    start_gradients = rescaled_note_number.copy()[:,:,0]
    end_gradients = rescaled_note_number.copy()[:,:,0]
    print("note number shapes:",note_number.shape,rescaled_note_number.shape)
    midi_wave[:] = 0
    start_gradients[:] = 1
    end_gradients[:] = 1
    for n,channel in enumerate(rescaled_note_number):
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if note[1] != channel[i-1][1] and channel[i][1] == channel[i+500][1] : ## every note start
                        wave_duration = 1
                        ind = 0
                        while True:
                            if i+ind >= channel.shape[0]-1 or (note[1] != channel[i+ind+1][1] and channel[i+ind+1][1] == channel[i+ind+500][1]):
                                break
                            wave_duration += 1
                            ind+=1
                            
                        freq = 440*(2**((channel[i+int(wave_duration/2)][0]-69)/12))
                        wave = make_wave(freq, wave_duration, 22050)
                        general_gradient_amt = 1800#int(wave_duration/gradient_fraction)
                        general_gradient = []
                        for g in range(0,general_gradient_amt):
                            general_gradient.append(g/general_gradient_amt)
                        for j,value in enumerate(wave):
                            
                            if midi_wave[n][i+j] != 0:
                                print("oof")
                            midi_wave[n][i+j]=value
                            try:
                                start_gradients[n][i+j] = general_gradient[j]
                                #if end_gradients[n][i+j] != 1:
                                #    print("oof")
                                end_gradients[n][i+(wave_duration-j)-1] = general_gradient[j]
                                #if start_gradients[n][i+(wave_duration-j)-1] != 1:
                                #    print("oof")
                            except Exception as e:
                                pass
                                
                            if i+j > last:
                                last = i+j
                except Exception as e:
                    print(i+ind)
                    print(ind)
                    print(channel.shape[0])
                    print(note[1])
                    print(channel[i+ind+1][1])
                    print(e)
                    print(last_start, i)
                    cont = input("...")
                    

    midi_wave = midi_wave[:][:last+1]
    actual_wave = np.zeros(midi_wave[0].shape[0])
    for n,channel in enumerate(midi_wave):
        if end_gradient:
            print("using end gradient")
            channel*=end_gradients[n]
        if start_gradient:
            print("using start gradient")
            channel*=start_gradients[n]
            print(start_gradients[n][0])
        actual_wave += channel
    return actual_wave/np.max(actual_wave), midi_wave, start_gradients, end_gradients
    

slide_window = 512


set_size = 512
pathes = []
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/0")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/2")
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning continuous both gradient with gradients"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
time_split = 512 ##milliseconds

midis = []
wavs = []
sets = 0
sets_ = []
start_index = 0
for set_num in range(0,3):
    path = pathes[set_num]
    print(path)
    ###loading in spectrograms-----------------------------------------------------------
    y = load_wave(path+"/wavs")
    y = y*0.1/np.max(y)
    wav_length = y.shape[0]
    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    f_wav,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
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
    '''if set_num == 0:
        graph = np.zeros(wav_specgram.shape)
        print(graph.shape)
        print(midi_graph.shape)
        rescale_factor = graph.shape[1]/midi_graph.shape[1]
        for c,channel in enumerate(midi_graph):
            for n,note in enumerate(channel):
                freq = note[0]#440*(2**((note[0]-69)/12))
                upper = 0
                for f,frq in enumerate(f_wav):
                    if frq > freq:
                        upper = f 
                        break
                upper_freq = f_wav[upper]
                lower_freq = f_wav[upper-1]
                freq_frac = (freq-lower_freq)/(upper_freq-lower_freq)
                graph[upper-1,int((n)*rescale_factor)] = 1-freq_frac ##lower
                graph[upper,int((n)*rescale_factor)] = freq_frac

        midi_specgram = graph.copy()
        print("graph shape:",graph.shape)
    else:
        graph = np.zeros(wav_specgram.shape)
        print(graph.shape)
        print(midi_graph.shape)
        rescale_factor = graph.shape[1]/midi_graph.shape[1]
        for c,channel in enumerate(midi_graph):
            for n,note in enumerate(channel):
                freq = 440*(2**((note[0]-69)/12))
                upper = 0
                for f,frq in enumerate(f_wav):
                    if frq > freq:
                        upper = f 
                        break
                upper_freq = f_wav[upper]
                lower_freq = f_wav[upper-1]
                freq_frac = (freq-lower_freq)/(upper_freq-lower_freq)
                graph[upper-1,int((n)*rescale_factor)] = 1-freq_frac ##lower
                graph[upper,int((n)*rescale_factor)] = freq_frac

        midi_specgram = graph.copy()
        print("graph shape:",graph.shape)  '''              

    if set_num == 1:
        one,complete_midi_graph,_,_ = note_number_to_wave(midi_graph)
        sd.play(one*0.1,22050)
        #sont = input(":")
        complete_midi_wave = np.zeros(complete_midi_graph.shape[1])
        for n,channel in enumerate(complete_midi_graph):
            complete_midi_wave += channel#*start_gradient_graph[n]*end_gradient_graph[n]


        complete_midi_wave = complete_midi_wave*0.1/np.max(complete_midi_wave) #sd.play(midi_wave,22050)
        midi_wave = complete_midi_wave
        #cont = input("...")

    elif set_num == 2:
        one,complete_midi_graph,_,_ = note_number_to_wave(midi_graph)
        sd.play(one*0.1,22050)
        #sont = input(":")
        complete_midi_wave = np.zeros(complete_midi_graph.shape[1])
        for n,channel in enumerate(complete_midi_graph):
            complete_midi_wave += channel#*start_gradient_graph[n]*end_gradient_graph[n]


        complete_midi_wave = complete_midi_wave*0.1/np.max(complete_midi_wave) #sd.play(midi_wave,22050)
        midi_wave = complete_midi_wave
        #cont = input("...")

    else:
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
    print(specgram.shape)
    for thing in f:
        print(thing)
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
    converted_back_wav = (converted_back_wav-1) + converted_back_midi
    '''
    converted_back_wav = np.transpose(timef_wav)
    arg_maxed = np.argmax(timef_wav,axis=-1)
    stacked = np.stack([np.zeros(arg_maxed.shape) for i in range(timef_wav.shape[-1])], axis=-1)
    for n,max_ in enumerate(arg_maxed):
        stacked[n,max_] = 1

    converted_back_wav = np.transpose(stacked)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(timef_midi[:1024])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(timef_wav[:1024])
    plt.show()
            
    print("arg shape",arg_maxed.shape,stacked.shape)
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

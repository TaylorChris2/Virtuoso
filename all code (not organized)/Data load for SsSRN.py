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
import copy
from scipy import signal
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
    
class hp:
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
        
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels =  128 # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = 512 # == hidden units of SSRN
    attention_win_size = 3

    # data
    data = "/data/private/voice/LJSpeech-1.0"
    # data = "/data/private/voice/kate"
    test_data = 'harvard_sentences.txt'
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS.
    max_N = 180 # Maximum number of characters.
    max_T = 256 # Maximum number of mel frames.

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/LJ01"
    sampledir = 'samples'
    B = 32 # batch size
    num_iterations = 2000000

def get_spectrograms(wave):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y = wave

    # Trimming
    #y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag



def load_spectrograms(wave):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    mel, mag = get_spectrograms(wave)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::hp.r, :]
    return mel, mag


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        print(i)
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array

def make_wave(freq, duration, sample_rate = 22050):
    wave = [i/((sample_rate/(2*np.pi))/freq) for i in range(0, int(duration))]
    wave = np.stack(wave)
    wave = np.cos(wave)
    '''
    sd.play(wave,sample_rate)
    cont = input("...")
    '''
    return wave


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
    


def load_wave(path):
    complete_wave = []
    file = 0
    first = False
    while True:
        try:
            
            wave_array = load_array(path+"/"+str(file)+".h5")
            first = True
            for moment in wave_array:
                complete_wave.append(moment)
            file+=1
        except:
            if first:
                break
            else:
                file+=1
    complete_wave = np.stack(complete_wave)
    return complete_wave

def load_graph(path):
    complete_graph = []
    for i in range(0, load_array(path+"/"+os.listdir(path)[0]).shape[0]):
        complete_graph.append([])
    file = 0
    first = False
    while True:
        try:
            array = load_array(path+"/"+str(file)+".h5")
            first = True
            for n,channel in enumerate(array):
                for moment in channel:
                    complete_graph[n].append(moment)
            file+=1
        except:
            if first:
                break
            else:
                file+=1
                
    complete_graph = np.stack(complete_graph)
    return complete_graph



def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram
    Args:
      mag: A numpy array of (T, 1+n_fft//2)
    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

slide_window = hp.max_T


set_size = 9238479128374
pathes = []
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/0")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/2")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/3")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/4")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/5")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/6")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/7")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/8")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/9")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/10")
pathes.append("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/11")
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning SRN"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
time_split = hp.max_T ##milliseconds

midis = []
wavs = []
sets = 0
sets_ = []
start_index = 0
for set_num in range(0,len(pathes)):
    path = pathes[set_num]
    print(path)
    ###loading in spectrograms-----------------------------------------------------------
    
    y = load_wave(path+"/wavs")

    y = y*0.1/np.max(y)

    mel, mag = load_spectrograms(y)
    print(mel.shape)

    converted = spectrogram2wav(mag)

    print(converted.shape)

    sd.play(converted,22050)

    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    f_wav,t,specgram = stft(converted,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    specgram = np.real(specgram)
    specgram[specgram < 0] = 0


    mel = librosa.amplitude_to_db(specgram, top_db=None)

    mel+=100
    mel/=100

    mel = mel.T
    print(mel.shape)




    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    f_wav,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    specgram = np.real(specgram)
    specgram[specgram < 0] = 0

    mag = librosa.amplitude_to_db(specgram, top_db=None)  # (T, 1+n_fft//2)
    mag+=100
    mag/=100

    mag = mag.T

    print(mel.shape,mag.shape,np.min(mag),np.max(mag),np.min(mel),np.max(mel))

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(mel[:4096])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mag[:4096])
    plt.show()




    sets+=1

    timef_midi = mel
    timef_wav = mag

    '''fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(mel[:512])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(mag[:512])
    plt.show()'''


 
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

    time_split_mag=time_split
    slide_window_mag=slide_window
    print(time_split_mag,slide_window)

        
    index = 0
    segments = []
    start = 0
    end = time_split_mag
    while True:
        segments.append(np.array(timef_wav[start:end]))
        start += slide_window_mag
        end += slide_window_mag
        if np.array(timef_wav[start:end]).shape[0] < time_split_mag:
            break
    if not delete_last:
        padding_amt = time_split_mag-segments[-1].shape[0]
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

    
new_indexes = []
for i in range(0,len(midis)):
    index = random.randint(0,len(midis)-1)
    while index in new_indexes:
        index = random.randint(0,len(midis)-1)
    new_indexes.append(index)

print(new_indexes)
print(len(midis),len(wavs))

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

import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import copy
from scipy import signal
import h5py
import os
import skimage.transform
from scipy.signal import stft

class hp:
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
        
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 128  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 100  # Number of inversion iterations
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
    max_T = 210 # Maximum number of mel frames.

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

def load_wave(path):
    complete_wave = []
    file = 0
    while True:
        try:
            wave_array = load_array(path+"/"+str(file)+".h5")    
            for moment in wave_array:
                complete_wave.append(moment)
            file+=1
        except Exception as e:
            print(e)
            break
    complete_wave = np.stack(complete_wave)
    return complete_wave

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
    #wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)

path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1"

y = load_wave(path+"/wavs")

mel1, mag = get_spectrograms(y)

sd.play(spectrogram2wav(mag),22050)
cont = input()

midi_graph = load_graph(path+"/midis/no gradient")
start_gradient_graph = load_graph(path+"/midis/start gradient graphs")       
end_gradient_graph = load_graph(path+"/midis/end gradient graphs")

one,complete_midi_graph,_,_ = note_number_to_wave(midi_graph)
sd.play(one*0.1,22050)
#sont = input(":")
complete_midi_wave = np.zeros(complete_midi_graph.shape[1])
for n,channel in enumerate(complete_midi_graph):
    complete_midi_wave += channel#*start_gradient_graph[n]*end_gradient_graph[n]



complete_midi_wave = complete_midi_wave*0.1/np.max(complete_midi_wave) #sd.play(midi_wave,22050)

mel2, mag = get_spectrograms(complete_midi_wave)


print(mel2.shape,mag.shape)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(mel1[:80])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(mel2[:80])
plt.show()





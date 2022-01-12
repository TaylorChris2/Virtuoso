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
import SimpleITK as sitk
import cv2
import wave
import scipy
from scipy.io.wavfile import write as save_wave
start_time = time.time()

class hp:
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
        
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.003125  # seconds
    frame_length = 0.0125  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 128 # Number of Mel banks to generate
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
    max_T = 512 # Maximum number of mel frames.

    # training scheme
    lr = 0.001 # Initial learning rate.
    logdir = "logdir/LJ01"
    sampledir = 'samples'
    B = 32 # batch size
    num_iterations = 2000000

def load_midi_violin(path):
    note_events = []
    mid = midi.read_midifile(path)
    ##getting only the note data
    for n,track in enumerate(mid):
        #print("\n\n\n\n\n\n")
        note_events.append([])
        for event in track:
            #print(event)
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
    print("only notes length:",len(only_notes))
    ##getting track length
    track_lengths = []
    for n,track in enumerate(only_notes):
        track_lengths.append(0)
        for event in track:
            track_lengths[n] += event.tick
    track_length = max(track_lengths)
    
    ##creating the actual track array and filling with empties
    track_array = []
    for i in range(0,len(only_notes)):
        track_array.append(np.zeros((track_length,2)))##one four channel list for pitch and one for articulation
    track_array = np.stack(track_array)
    ##filling in the track array with real note data
    for s,slot in enumerate(only_notes):
        current_tick = 0
        note_num = 1
        for n,event in enumerate(only_notes[s]):
            current_tick += event.tick
            if event.data[1] > 0:##every note start
                for i in range(current_tick,current_tick+only_notes[s][n+1].tick):
                    track_array[s][i][0] = event.data[0]
                    track_array[s][i][1] = note_num
                note_num += 1
                    
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
    wave = np.sin(wave)
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


           
def note_number_to_wave(note_number, gradient_fraction=3, end_gradient = True, start_gradient = False, rescale_factor=1):
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
            if note[0]>=1: ## pitch
                try:
                    if i+500 < channel.shape[0] and note[1] != channel[i-1][1] and channel[i][1] == channel[i+500][1] and channel[i][1] != channel[i-500][1]: ## every note start
                        #print(channel[i-100:i+100])
                        
                        wave_duration = 1
                        ind = 0
                        while True:
                            if i+ind >= channel.shape[0]-1 or (i+ind+500 < channel.shape[0] and note[1] != channel[i+ind+1][1] and channel[i+ind+1][1] == channel[i+ind+500][1] and channel[i+ind+1][1] != note[1]):
                                break
                            wave_duration += 1
                            ind+=1

                        print(n,i,channel[i+int(wave_duration/2)][0])
                            
                        freq = 440*(2**((channel[i+int(wave_duration/2)][0]-69)/12))
                        wave = make_wave(freq, wave_duration, 22050)
                        general_gradient_amt = int(wave_duration/gradient_fraction)
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
    #sd.play(midi_wave[1],22050)
    actual_wave = np.zeros(midi_wave[0].shape[0])
    print(np.max(actual_wave))

    ovlp = actual_wave.copy()
    ovlp[:] = 1
    print("bur:",np.max(ovlp))
    
    for n,channel in enumerate(midi_wave):
        print("channel max",np.max(channel),n)
        for x,moment in enumerate(channel):
            if n>0:
                pass#print(moment,actual_wave[x],n)
            if moment!=0 and actual_wave[x]!=0:
                #print(moment,actual_wave[x],n,x)
                ovlp[x] += 1
                
                
                #actual_wave[x]/=2
        if end_gradient:
            print("using end gradient")
            channel*=end_gradients[n]
        if start_gradient:
            print("using start gradient")
            channel*=start_gradients[n]
            print(start_gradients[n][0])

        print("channel max:",np.max(channel))

        actual_wave+=channel



    print("ur stoopid:",np.min(ovlp),np.max(ovlp),np.max(actual_wave))


                                 

    actual_wave/=ovlp
    if np.max(actual_wave) > 0:
        actual_wave/=np.max(actual_wave)
    return actual_wave, rescaled_note_number, start_gradients, end_gradients

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

def convex_border(array, thickness):
    contour_only = array.copy()
    binary = array.copy()

    contour_only[:] = 0
    
    binary[:] = 0
    binary[array > 0] = 255

    for i in range(0,4):
        binary[binary > 0] = 255
        contours, _ = cv2.findContours(binary.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #hull = cv2.convexHull(biggest_contour)
        cv2.drawContours(binary, contours, -1, 200, thickness)
    
    binary = (binary>0).astype("float64")
    return binary

def fill_holes_binary(array, sense):
    binary = array.copy()

    binary_original = array.copy()

    binary_original[:] = 0
    binary_original[array > 0] = 1
    
    binary[:] = 0
    binary[array == 0] = 1




    touching_structure_2d = [[0,1,0],
                             [1,1,1],
                             [0,1,0]]



    markers, num_features = scipy.ndimage.measurements.label(binary,touching_structure_2d)
    omit = markers[-1,0]
    flat = markers.ravel()
    binc = np.bincount(flat)
    binc_not = np.bincount(flat[flat == omit])
    noise_idx2 = np.where(binc > sense)
    noise_idx1 = np.where(binc == np.max(binc_not))
    
    mask1 = np.isin(markers, noise_idx1)
    mask2 = np.isin(markers, noise_idx2)
    
    binary[mask1] = 0
    binary[mask2] = 0

    binary_original[binary == 1] = 1

        
    return binary_original
def load_array(path):
    h5f = h5py.File(path,'r')
    array = h5f['all_data'][:]
    h5f.close()
    return array

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
        except Exception as e:
            print(e)
            break
    complete_graph = np.stack(complete_graph)
    return complete_graph

    

def note_number_to_wave(note_number, gradient_fraction=3, end_gradient = True, start_gradient = False, rescale_factor=1):
    last = 0
    rescaled_note_number = np.round(note_number.copy())#skimage.transform.rescale(note_number, (1, rescale_factor, 1))
    midi_wave = rescaled_note_number.copy()[:,:,0]
    start_gradients = rescaled_note_number.copy()[:,:,0]
    end_gradients = rescaled_note_number.copy()[:,:,0]
    print("note number shapes:",note_number.shape,rescaled_note_number.shape)
    midi_wave[:] = 0
    start_gradients[:] = 1
    end_gradients[:] = 1
    for n,channel in enumerate(rescaled_note_number):
        for i,note in enumerate(channel):
            if note[0]>=1: ## pitch
                try:
                    if i+500 < channel.shape[0] and note[1] != channel[i-1][1] and channel[i][1] == channel[i+500][1] and channel[i][1] != channel[i-500][1]: ## every note start
                        #print(channel[i-100:i+100])
                        
                        wave_duration = 1
                        ind = 0
                        while True:
                            if i+ind >= channel.shape[0]-1 or (i+ind+500 < channel.shape[0] and note[1] != channel[i+ind+1][1] and channel[i+ind+1][1] == channel[i+ind+500][1] and channel[i+ind+1][1] != note[1]):
                                break
                            wave_duration += 1
                            ind+=1

                        print(n,i,channel[i+int(wave_duration/2)][0])
                            
                        freq = 440*(2**((channel[i+int(wave_duration/2)][0]-69)/12))
                        wave = make_wave(freq, wave_duration, 22050)
                        general_gradient_amt = int(wave_duration/gradient_fraction)
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
    #sd.play(midi_wave[1],22050)
    actual_wave = np.zeros(midi_wave[0].shape[0])
    print(np.max(actual_wave))

    ovlp = actual_wave.copy()
    ovlp[:] = 1
    print("bur:",np.max(ovlp))
    
    for n,channel in enumerate(midi_wave):
        print("channel max",np.max(channel),n)
        for x,moment in enumerate(channel):
            if n>0:
                pass#print(moment,actual_wave[x],n)
            if moment!=0 and actual_wave[x]!=0:
                #print(moment,actual_wave[x],n,x)
                ovlp[x] += 1
                
                
                #actual_wave[x]/=2
        if end_gradient:
            print("using end gradient")
            channel*=end_gradients[n]
        if start_gradient:
            print("using start gradient")
            channel*=start_gradients[n]
            print(start_gradients[n][0])

        print("channel max:",np.max(channel))

        actual_wave+=channel



    print("ur stoopid:",np.min(ovlp),np.max(ovlp),np.max(actual_wave))


                                 

    actual_wave/=ovlp
    return actual_wave/np.max(actual_wave), rescaled_note_number, start_gradients, end_gradients


def writeout(path,sr,wave):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    save_wave(path,sr,wave)

           
sampling_size = 44100
in_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/autosyncing final"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/loaded arrays testing final edited fully"

bad = [5,11,3,12,16,17,19,10,9,2]
while True:
    start = 0
    sets = [str(i)for i in sorted([int(x) for x in os.listdir(in_path) if int(x) not in bad])]
    print(sets)
    for i in range(start,len(sets)):
        set_ = sets[i]
        print(set_)
        path = in_path+"/"+set_
        complete_wav = load_wave(path+"/wavs")

        complete_midi_graph = load_graph(path+"/midis/no gradient")
        complete_start_gradient_graph = load_graph(path+"/midis/start gradient graphs")
        complete_end_gradient_graph = load_graph(path+"/midis/end gradient graphs")

        midi_wave, midi_graph, start_gradient_graph, end_gradient_graph = note_number_to_wave(complete_midi_graph, end_gradient = True, start_gradient = False, rescale_factor = 1)
        midi_wave /= np.max(midi_wave)
        complete_wav /= np.max(complete_wav)
        added = midi_wave[:complete_wav.shape[0]]+complete_wav[:midi_wave.shape[0]]
        added/=np.max(added)
        #sd.play(added,22050)
        if True:#input("gucc?:")!="n":
            writeout(output+"/"+set_+"-midi.wav",22050,np.array(midi_wave*32767, dtype=np.int16))
            writeout(output+"/"+set_+"-wav.wav",22050,np.array(complete_wav*32767, dtype=np.int16))
            writeout(output+"/"+set_+"-added.wav",22050,np.array(added*32767, dtype=np.int16))

            print(time.time()-start_time)




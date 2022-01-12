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
start_time = time.time()

class hp:
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
        
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 4096  # fft points (samples)
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


           
def note_number_to_wave(note_number, gradient_fraction=3, end_gradient = True, start_gradient = True, rescale_factor=1):
    last = 0
    rescaled_note_number = skimage.transform.rescale(note_number, (1, rescale_factor, 1))
    midi_wave = rescaled_note_number.copy()[:,:,0]
    start_gradients = rescaled_note_number.copy()[:,:,0]
    end_gradients = rescaled_note_number.copy()[:,:,0]
    print("note number shapes:",note_number.shape,rescaled_note_number.shape)
    midi_wave[:] = 0
    start_gradients[:] = 1
    end_gradients[:] = 1
    first = False
    for n,channel in enumerate(rescaled_note_number):
        for i,note in enumerate(channel):
            if note[0]>0: ## pitch
                try:
                    if not first or (i+1<channel.shape[0] and note[1] != channel[i-1][1] and channel[i][0] == channel[i+1][0]): ## every note start
                        first = True
                        wave_duration = 1
                        ind = 0
                        while True:
                            try:
                                if i+ind >= channel.shape[0]-1 or (note[1] != channel[i+ind+1][1] and channel[i+ind+1][0] == channel[i+ind+2][0]):
                                    break
                            except Exception as e:
                                print(e)
                                break
                            wave_duration += 1
                            ind+=1
                            
                        freq = 440*(2**((channel[i+int(wave_duration/2)][0]-69)/12))
                        wave = make_wave(freq, wave_duration, 22050)
                        general_gradient_amt = int(wave_duration/gradient_fraction)
                        general_gradient = []
                        for g in range(0,general_gradient_amt):
                            general_gradient.append(g/general_gradient_amt)
                        for j,value in enumerate(wave):
                            
                            if midi_wave[n][i+j] != 0:
                                pass
                                #print("oof")
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
    
    return actual_wave/np.max(actual_wave), rescaled_note_number, start_gradients, end_gradients

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
    mag = mag[::hp.r, :]
    return mel, mag

               
sampling_size = 44100
path ="C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut dat 10"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/autosyncing/0"
frequency_clip_midi = 512 ##amount of frequencies to be included
frequency_clip_wav = 512 ##amount of frequencies to be included
Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000


midi_durations = []

sets = []

start_midi_tick = 0
start_wav_tick = 0
save_index = 0

current_saved = 0
for set_ in os.listdir(path):
    sets.append(set_)
print(sets)

start_index = 0

for set_num in range(start_index, len(sets)):
    print("\n"+sets[set_num]+"\n")
    found_wav = False
    found_mid = False
    for file in os.listdir(os.path.join(path,sets[set_num])):
        if file.endswith(".wav") and not found_wav:
            orig_wav,sr = librosa.load(os.path.join(os.path.join(path,sets[set_num]), file))
            orig_wav/=np.max(orig_wav)
            for n,moment in enumerate(orig_wav):
                if abs(moment)>0.025:
                    print("first wav:", n)
                    start_wav_tick = n
                    break
                
            wav_length = orig_wav.shape[0]
            found_wav = True
            print("Loaded wave file.")
            
        elif file.endswith("mid") and not found_mid:
            midi_array = load_midi_violin(os.path.join(os.path.join(path,sets[set_num]), file))
            print(midi_array.shape)
            first = midi_array.shape[1]
            for channel in midi_array:
                found_first = False
                for n,note in enumerate(channel):
                    #print(note)
                    if max(note) > 0:
                        if n < first:
                            first = n
                            start_midi_tick = n
            midi_array = midi_array[:,first:]
            print("first:",first)

            print(midi_array.shape)
            found_mid = True
            print("Loaded midi file.")
    length_factor = int(wav_length/midi_array.shape[1])
    print("\n\nlength factor:",length_factor,"\n\n")
    
    wav_tick = start_wav_tick
    midi_tick = start_midi_tick
    start_midi_tick = 0
    start_wav_tick = 0
    while True:
        print("midi tick:",midi_tick,"wav tick:",wav_tick)
        #good = float(input("enter a factor:"))
        metrics = []
        wavs = []
        midi_graphs = []
        start_gradient_graphs = []
        end_gradient_graphs = []

        for good in range(70,71):
            good = 0.5#good/100
            real_length = round(sampling_size/float(good))
            actual_midi_tick = int(round(midi_tick*float(good)))
            print(real_length)
            print(actual_midi_tick)
            current_array = midi_array[:,int(midi_tick/length_factor):int((midi_tick/length_factor)+(real_length/length_factor))+1]
            midi_wave, midi_graph, start_gradient_graph, end_gradient_graph = note_number_to_wave(current_array, end_gradient = True, start_gradient = False, rescale_factor = length_factor*(sampling_size/real_length))
            _,midi_specgram = load_spectrograms(midi_wave)
            
            midi_graph = midi_graph[:,:sampling_size]
            print("midi graph shape:",midi_graph.shape)
            start_gradient_graph = start_gradient_graph[:,:sampling_size]
            print("start gradient graph shape:",start_gradient_graph.shape)
            end_gradient_graph = end_gradient_graph[:,:sampling_size]
            print("end gradient graph shape:",end_gradient_graph.shape)
            
            midi_wave = midi_wave[:sampling_size]
            midi_wave = midi_wave*0.1/np.max(midi_wave)
            if midi_wave.shape[0] < sampling_size:
                padding_amt = sampling_size-midi_wave.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in midi_wave:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                midi_wave = np.stack(padded)
            print(midi_wave.shape)

            wav = orig_wav[wav_tick:wav_tick+sampling_size]
            if wav.shape[0] < sampling_size:
                padding_amt = sampling_size-wav.shape[0]
                padding = np.zeros(padding_amt)
                padded = []
                for time_ in wav:
                    padded.append(time_)
                for pad in padding:
                    padded.append(pad)
                wav = np.stack(padded)
            _,wav_specgram = load_spectrograms(wav)
            wav_specgram1,_ = load_spectrograms(wav)

            min_blub = 1280
            max_blub = 0
            for x,thing in enumerate(midi_specgram):
                for j, pixel in enumerate(thing):
                    if pixel > 0.5:
                        if j>max_blub:
                            max_blub = j
                        elif j<min_blub:
                            min_blub = j
            wav_specgram = wav_specgram[:,min_blub:max_blub]
            midi_specgram = midi_specgram[:,min_blub:max_blub]

            best = 0
            metric = 0
            for i in range(70,130):
                test = midi_specgram.copy()
                test/=np.max(test)
                test = skimage.transform.rescale(test,((2*i/100),1))
                print(test.shape,wav_specgram.shape)
                test = test[:wav_specgram.shape[0]]
                test_wav = wav_specgram.copy()
                
                test_wav/=np.max(test_wav)
                #test = (test>0.8).astype("float64")
                #test_wav = (test_wav > 0.5).astype("float64")


                
                '''for t,thing in enumerate(test):
                    if np.max(thing) > 0.5:
                        test[t,:] = 1
                    else:
                        test[t,:] = 0
                for t,thing in enumerate(test_wav[:,min_blub:max_blub]):
                    if np.mean(thing) > 0.4:
                        test_wav[t,:] = 1
                    else:
                        test_wav[t,:] = 0
                
                for n,thing in enumerate(test):
                    print(np.argmax(thing),np.argmax(test_wav[n]))'''
                
                err = (np.argmax(test,axis=-1)-np.argmax(test_wav,axis=-1))**2
                #err[test!=0] = 0
                mse = np.sum(err)
                
                fig = plt.figure()
                fig.subplots_adjust(hspace=0.4, wspace=0.4)
                ax = fig.add_subplot(1, 3, 1)
                ax.imshow(test)
                #ax = fig.add_subplot(1, 3, 2)
                #ax.imshow(err)
                ax = fig.add_subplot(1, 3, 3)
                ax.imshow(test_wav)
                plt.show()
                print(np.max((test-test_wav)**2),np.min((test-test_wav)**2))
                print(mse)
                print(i)
                if i==70:
                    metric = mse
                    best = i/50
                elif mse<metric:
                    metric = mse
                    best = i/50
            print("best scale:",best)
            #best = 1.82

            midi_wave, midi_graph, start_gradient_graph, end_gradient_graph = note_number_to_wave(current_array, end_gradient = True, start_gradient = False, rescale_factor = length_factor*(sampling_size/real_length)*best)
            
            midi_wave = midi_wave[:sampling_size]
            sd.play(midi_wave+wav,22050)
            midi_specgram, mag = load_spectrograms(midi_wave)
            midi_graph = midi_graph[:,:sampling_size]
            print("midi graph shape:",midi_graph.shape)
            start_gradient_graph = start_gradient_graph[:,:sampling_size]
            print("start gradient graph shape:",start_gradient_graph.shape)
            end_gradient_graph = end_gradient_graph[:,:sampling_size]
            print("end gradient graph shape:",end_gradient_graph.shape)

            '''fig = plt.figure()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(midi_specgram)
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(wav_specgram1)
            plt.show()'''


            save_array(wav, output+"/wavs/"+str(save_index)+".h5")
            save_array(midi_graph, output+"/midis/no gradient/"+str(save_index)+".h5")
            save_array(start_gradient_graph, output+"/midis/start gradient graphs/"+str(save_index)+".h5")
            save_array(end_gradient_graph, output+"/midis/end gradient graphs/"+str(save_index)+".h5")
            save_index+=1
            wav_tick += sampling_size
            midi_durations.append(real_length)
            midi_tick += real_length




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
    return mel, mag

def register(fixed_image, moving_image, orig, transform = None):
    if transform is None:
        resamples = []
        metrics = []
        transforms = []
        for i in range (1,10):
            ImageSamplingPercentage = 1
            initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.ScaleTransform(2,(1, 0)), sitk.CenteredTransformInitializerFilter.MOMENTS)
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetMetricAsMeanSquares()
            registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
            registration_method.SetMetricSamplingPercentage(float(ImageSamplingPercentage)/100)
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(learningRate=0.001, numberOfIterations=10**5, convergenceMinimumValue=1e-6, convergenceWindowSize=100) #Once
            registration_method.SetOptimizerScalesFromPhysicalShift() 
            registration_method.SetInitialTransform(initial_transform)

            transform = registration_method.Execute(fixed_image, moving_image)
            #print(transform)
            print("number:",i)
            print(registration_method.GetMetricValue())
            metrics.append(registration_method.GetMetricValue())
            resamples.append(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID()))
            transforms.append(transform)
        print(np.min(metrics))
        return sitk.GetArrayFromImage(resamples[metrics.index(np.min(metrics))]),transforms[metrics.index(np.min(metrics))]
    else:
        return sitk.GetArrayFromImage(sitk.Resample(orig, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())),transform

               
sampling_size = 44100
path = path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 13"
output = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/10"
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

## for preludio p3
#start_midi_tick = 226086
#start_wav_tick = 248600#28100
#save_index = 5

## for p2 chaconne/alle
##start_midi_tick = 1352742#99000
##start_wav_tick = 1646700#15000
##save_index = 37
##

##for p1 6 double
#start_midi_tick = 843054
#start_wav_tick = 803800#10000
#save_index = 18

##for p2 gigue
#start_midi_tick = 308125
#start_wav_tick = 427200#30300
#save_index = 18

##for p1 2 double
#start_midi_tick = 731527
#start_wav_tick = 715600#10000
#save_index = 32

##for s2 adante
#start_midi_tick = #6000
#start_wav_tick = #15000
#save_index = 0

##for s3 adagio
#start_midi_tick = 1178465
#start_wav_tick = 1168600#22000
#save_index = 27

##for p3 menuet I
#start_midi_tick =282102
#start_wav_tick = 584200#55000
#save_index = 13

##for S1 presto
start_midi_tick = 266381
start_wav_tick = 289600#25000
save_index = 7


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
            #sd.play(orig_wav,22050)
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
                    if max(note) > 0 and not found_first:
                        if n < first:
                            first = n
            midi_array = midi_array[:,first:]
            print(first)
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
        good = float(input("enter a factor:"))
        while good != "y":
            real_length = round(sampling_size/float(good))
            actual_midi_tick = int(round(midi_tick*float(good)))
            print(real_length)
            print(actual_midi_tick)
            current_array = midi_array[:,int(midi_tick/length_factor):]#int((midi_tick/length_factor)+(real_length/length_factor))+1]
            midi_wave, midi_graph, start_gradient_graph, end_gradient_graph = note_number_to_wave(current_array, end_gradient = True, start_gradient = False, rescale_factor = length_factor*(sampling_size/real_length))
            midi_specgram, mag = load_spectrograms(midi_wave)
            
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
            wav_specgram, mag = load_spectrograms(wav)
            
            '''blub = np.argmax(midi_specgram[:1600],axis=-1)
            for n,thing in enumerate(blub):
                print(thing)
                midi_specgram[n][:] = 0
                midi_specgram[n][thing] = 1
            blub = np.argmax(wav_specgram[:1600],axis=-1)
            for n,thing in enumerate(blub):
                wav_specgram[n][:] = 0
                wav_specgram[n][thing] = 1'''


            print(wav_specgram.shape,midi_specgram.shape)
            uno = sitk.GetImageFromArray(wav_specgram)
            dos = sitk.GetImageFromArray(midi_specgram)

            
            blub,_ = register(uno,dos,dos)
            
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(blub)
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(wav_specgram)
            plt.show()
            
            added = midi_wave+(0.5*wav)
            sd.play(added,11025)
            good = input("did it sound good? (enter next factor for no) (yes for good) (b to go back one) (r to replay):")
            while good == "r":
                speed = float(input("what speed do you wanted it played at?:"))
                if speed < 0:
                    sd.play(midi_wave,-int(22050/speed))
                else:
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


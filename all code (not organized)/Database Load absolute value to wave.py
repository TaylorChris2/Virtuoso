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
        track_array.append([[0.,0.,0.,0.],[1.,1.,1.,1.],[0.,0.,0.,0.]])##one four channel list for pitch and one for articulation
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
                    if track_array[current_tick][2][slot] == 0:
                        track_array[current_tick][2][slot] = track[n+1].tick
                        break
                for i in range(0,int(track[n+1].tick/4)):
                    track_array[current_tick+i][1][working_slot] = i/int(track[n+1].tick/4)
                    track_array[current_tick+track[n+1].tick-i-1][1][working_slot] = i/int(track[n+1].tick/4)
                    
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
    while True:
        try:
            if os.path.exists(path):
                os.remove(path)

            hdf5_store = h5py.File(path, "a")
            hdf5_store.create_dataset("all_data", data = array, compression="gzip")
            break
        except:
            pass

def midi_2_specgram(midi, length_factor):
    Fs = 22050
    N = 2048
    w = np.hamming(N)
    ov = N - Fs // 1000
    midi_wave = np.zeros((4,int(midi.shape[0]*length_factor)))
    articulation_factors = midi[:,1]
    articulation_factors = skimage.transform.rescale(articulation_factors, (length_factor, 1))
    print("articulation factors shape:",articulation_factors.shape)
    print("specgram shape:",midi_wave.shape)
    for channel in range(0,4):
        print("channel:",channel)
        for i,note in enumerate(midi):
            if note[0,channel]>0: ## pitch
                try:
                    if note[2,channel] > 0: ## every note start
                        print(i,note[2,channel])
                        freq = 440*(2**((note[0,channel]-69)/12))              
                        wave = make_wave(freq, note[2,channel]*length_factor, 22050)
                        for j,value in enumerate(wave):
                            midi_wave[channel,int(i*length_factor)+j]=wave[j]#*articulation_factors[int(i*length_factor)+j,channel]
                except Exception as e:
                    print(e)
                    print(last_start, i)
                    cont = input("...")
    _,_,first_channel = stft(midi_wave[0],nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
    specgram = np.zeros((first_channel.shape[0],first_channel.shape[1]))
    for channel in midi_wave:
        _,_,channel_specgram = stft(channel,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        Smag, p = librosa.magphase(channel_specgram)
        channel_specgram = librosa.amplitude_to_db(Smag, top_db=None)
        specgram += channel_specgram
    return specgram
            


set_size = 48
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data"
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning"
frequency_clip = 1024 ##amount of frequencies to be included
time_split = 8192 ##milliseconds

midis = []
wavs = []
sets = 0
sets_ = []
start_index = 0
for set_ in os.listdir(path):
    sets_.append(set_)
print(sets_)
for set_num in range(start_index, len(sets_)):
    print("\n"+sets_[set_num]+"\n")
    found_wav = False
    found_mid = False
    for file in os.listdir(os.path.join(path,sets_[set_num])):
        if file.endswith(".wav") and not found_wav:
            y,sr = librosa.load(os.path.join(os.path.join(path,sets_[set_num]), file))
            wav_length = y.shape[0]
            Fs = 22050
            N = 2048
            w = np.hamming(N)
            ov = N - Fs // 1000
            f,t,specgram = stft(y,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
            specgram = np.real(specgram)
            specgram[specgram < 0] = 0
            
            wav_specgram = []
            for i in range(0,frequency_clip):
                wav_specgram.append(specgram[i])
            wav_specgram = np.stack(wav_specgram)
            wav_specgram = wav_specgram/np.max(wav_specgram)
            wav_specgram = np.log10(wav_specgram)
            wav_specgram += 10
            wav_specgram[wav_specgram < 0] = 0
            wav_specgram = wav_specgram/10.1

            '''
            ##test for converting back and playing
            converted_back = wav_specgram*10.1
            converted_back-=10
            converted_back = 10**converted_back
            converted_back = converted_back*0.1/np.max(converted_back)
            decoded = []
            for freq in converted_back:
                decoded.append(freq)
            decoded.append(np.zeros(converted_back.shape[1]))
            decoded = np.stack(decoded)

            

            
            t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
            cont = "a"
            while cont == "a":
                sd.play(back,22050)
                cont = input("...")
                sd.play(y,22050)
                cont = input("...")
            '''
            
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
            found_wav = True
            print("Loaded wave file.")
            
        elif file.endswith("mid") and not found_mid:
            midi_array = load_midi_violin(os.path.join(os.path.join(path,sets_[set_num]), file))
            print("1st channel max:",np.max(midi_array[:,0,0]))
            print("2nd channel max:",np.max(midi_array[:,0,1]))
            print("3rd channel max:",np.max(midi_array[:,0,2]))
            print("4th channel max:",np.max(midi_array[:,0,3]))
            print(midi_array.shape)
            found_mid = True
            print("Loaded midi file.")

    if not found_wav or not found_mid:
        print("Data incomplete. Failed to load: " + os.path.join(path,sets_[set_num]))
    else:
        sets+=1
        rescale_factor = (wav_specgram.shape[1]/midi_array.shape[0])
        length_factor = (wav_length/midi_array.shape[0])
        specgram = midi_2_specgram(midi_array, length_factor)
        specgram[specgram < 0] = 0
        
        '''
        t,back = istft(specgram,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        sd.play(back,22050)
        cont = input("...")
        '''

        midi_specgram = specgram.copy()
        """
        midi_specgram = []
        for i in range(0,frequency_clip):
            midi_specgram.append(specgram[i])
        midi_specgram = np.stack(midi_specgram)
        """

        #midi_specgram = librosa.amplitude_to_db(midi_specgram)

        print("midi shape:",midi_specgram.shape)

        print(np.max(midi_specgram))
        print(np.min(midi_specgram))

        
        timef_midi = np.transpose(midi_specgram)
        timef_wav = np.transpose(wav_specgram)

        #timef_wav = timef_wav-timef_midi
        #timef_wav += 1
        timef_wav[timef_midi == 0] = 0
        print("specgram shapes:", timef_midi.shape,timef_wav.shape)
        print(np.max(timef_wav))
        print(np.min(timef_wav))

        print("Converted to spectrogram.")
        delete_last = False


        print("Split wav spectrograms.")
        
        index = 0
        segments = []
        while True:
            start = index*time_split
            end = (index+1)*time_split
            if np.array(timef_midi[start:end]).shape[0] == 0:
                break
            segments.append(np.array(timef_midi[start:end]))
            index += 1
        ##padding the ending
        if segments[-1].shape[0] > 3000:
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
        while True:
            start = index*time_split
            end = (index+1)*time_split
            if np.array(timef_wav[start:end]).shape[0] == 0:
                break
            segments.append(np.array(timef_wav[start:end]))
            index += 1
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
        '''
        decoded = []
        converted_back = np.transpose(timef_midi)*10.1
        converted_back-=10
        converted_back = 10**converted_back
        converted_back = converted_back*0.1/np.max(converted_back)
        decoded = []
        for freq in converted_back:
            decoded.append(freq)
        decoded.append(np.zeros(converted_back.shape[1]))
        decoded = np.stack(decoded)
        print(decoded.shape)
        t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        print(back[-1])
        sd.play(back,22050)
        time.sleep(9)

        converted_back = np.transpose(timef_wav)*10.1
        converted_back-=10
        converted_back = 10**converted_back
        converted_back = converted_back*0.1/np.max(converted_back)
        decoded = []
        for freq in converted_back:
            decoded.append(freq)
        decoded.append(np.zeros(converted_back.shape[1]))
        decoded = np.stack(decoded)
        t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
        print(back[-1])
        sd.play(back,22050)
        '''
        
print("Loaded in" ,len(midis),len(wavs), "sets from", sets, "folders in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")
midi_sets, wav_sets = seperate_sets(midis, wavs, set_size)

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

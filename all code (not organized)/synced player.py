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


def make_wave(freq, duration, sample_rate = 22050):
    wave = [i/((sample_rate/(2*np.pi))/freq) for i in range(0, int(duration))]
    wave = np.stack(wave)
    wave = np.cos(wave)
    '''
    sd.play(wave,sample_rate)
    cont = input("...")
    '''
    return wave


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





path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/autosyncing final/8"

complete_wav = load_wave(path+"/wavs")

complete_midi_graph = load_graph(path+"/midis/no gradient")


midi_wave, midi_graph, start_gradient_graph, end_gradient_graph = note_number_to_wave(complete_midi_graph, end_gradient = True, start_gradient = False, rescale_factor = 1)
sd.play(midi_wave*0.1,22050)
#sont = input(":")

complete_midi_wave =  midi_wave.copy()

complete_midi_wave = complete_midi_wave*0.1/np.max(complete_midi_wave)
complete_wav = complete_wav*0.1/np.max(complete_wav)

added = complete_midi_wave[:complete_wav.shape[0]]+complete_wav[:complete_midi_wave.shape[0]]
while True:
    sd.play(added,22050)
    cont = input()


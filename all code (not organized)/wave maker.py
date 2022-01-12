from Virtuoso_utils.utils import *
import os
import re
import time
import numpy as np
import midi
import sounddevice as sd 
from scipy.signal import istft
from scipy.signal import stft
import librosa
import librosa.display
import skimage.transform
import h5py
import matplotlib.pyplot as plt
import random
from scipy.special import expit as sigmoid
import tensorflow.keras as keras
import tensorflow as tf
import copy
from scipy import signal
from scipy.io.wavfile import write as writeout
import cv2
from pysndfx import AudioEffectsChain

fx = (
    AudioEffectsChain()
    .reverb()
)

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


save_folder = "C:/Users/JiangQin/Documents/c++/Grid-20210101T231410Z-001/build-Grid-Desktop_Qt_5_15_2_MSVC2015_64bit-Debug/notes"

for i in range(0,97):
    print(i)
    freq = 440*(2**((i-69)/12))
    wave = make_wave(freq,2205)
    
    
    for j in range(0,1000):
        wave[j]*=j/1000
        wave[-j]*=j/1000
    #sd.play(wave,22050)
    #cont = input()
    wave*=0.2
    writeout(save_folder+"/"+str(i)+".wav",22050,np.array(wave*32767, dtype=np.int16))
    


import sounddevice as sd
import numpy as np
from scipy.signal import istft
from scipy.signal import stft
import skimage.transform

Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000

def make_wave(freq, duration, sample_rate = 22050):
    wave = []
    for i in range(0,duration*sample_rate):
        wave.append(i/((sample_rate/(2*np.pi))/freq))

    wave = np.cos(np.stack(wave))
    return wave


            
A = make_wave(440, 10)
_,_,A_spec = stft(A,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
timef_spec = np.transpose(A_spec)
for moment in timef_spec:
    print(np.where(moment == np.max(moment)))
A_spec = np.real(A_spec)
A_spec[A_spec < 0] = 0
#A_spec = skimage.transform.rescale(A_spec, (1,0.96))
print(A_spec.shape)

t,back = istft(A_spec,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
#back = skimage.transform.rescale(back, 0.5)
sd.play(back, 22050)
cont = input("...")
sd.play(chord_non_spec*0.1/np.max(chord_non_spec), 22050)

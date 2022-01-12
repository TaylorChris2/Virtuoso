from data_load import get_batch
from utils import *
import sounddevice as sd
from hyperparams import Hyperparams as hp

y,_ = librosa.load("/home/jiangl/Documents/python/Music Composition Project/Music Files/Violin/bach partita 1/audio files/J.S. Bach Partita for Violin Solo No. 1 in B Minor, BWV 1002 - 3. Courante.mp3")

print(len(y))

sd.play(y, 22050)


for thing in mel:
    print(thing)
L, mels, mags, fnames, num_batch = get_batch()

print(L.shape)
print(mels.shape)
print(mags.shape)

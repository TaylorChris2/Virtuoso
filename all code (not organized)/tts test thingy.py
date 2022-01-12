import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import copy
from scipy import signal
from scipy.signal import istft
from scipy.signal import stft
import cv2

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

wave = librosa.load("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data 5/bla/Johann Sebastian Bach - Chaconne, Partita No. 2 BWV 1004   Hilary Hahn.wav")[0]

mel, mag = load_spectrograms(wave[:220500])
print(mel.shape,mag.shape)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(mel[:80])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(mag[:1024])
plt.show()
mag1 = cv2.blur(mag,(3,3))

converted1 = spectrogram2wav(mag1)
converted = spectrogram2wav(mag)


while True:
    sd.play(converted,22050)
    cont = input(":")
    sd.play(converted1,22050)
    cont = input(":")
    sd.play(wave[:220500],22050)
    cont = input(":")

cont = input()

Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f_wav,t,specgram = stft(converted,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
specgram = np.real(specgram)
specgram[specgram < 0] = 0


specgram_converted = librosa.amplitude_to_db(specgram, top_db=None)

specgram_converted = cv2.blur(specgram_converted,(3,3))

decoded = librosa.db_to_amplitude(specgram_converted)
t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)

sd.play(back/np.max(back),22050)

cont = input()



Fs = 22050
N = 2048
w = np.hamming(N)
ov = N - Fs // 1000
f_wav,t,specgram = stft(wave[:220500],nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)
specgram = np.real(specgram)
specgram[specgram < 0] = 0


specgram_wav = librosa.amplitude_to_db(specgram, top_db=None)

decoded = librosa.db_to_amplitude(specgram_wav)
t,back = istft(decoded,nfft=N,fs=Fs,window=w,nperseg=None,noverlap=ov)

sd.play(back,22050)



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(specgram_converted[:,1024:2048])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(specgram_wav[:,1024:2048])
plt.show()



sd.play(spectrogram2wav(mag),22050)


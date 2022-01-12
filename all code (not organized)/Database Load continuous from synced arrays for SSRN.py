import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import copy
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import h5py

def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           activation_fn=None,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      dropout_rate: A float of [0, 1].
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    if padding.lower() == "causal":
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list()[-1]

    params = {"filters": filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": "he_normal"}

    tensor = tf.keras.layers.Conv1D(**params)(inputs)
    tensor = normalize(tensor)
    if activation_fn is not None:
        tensor = activation_fn(tensor)

    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor

def hc(inputs,
       filters=None,
       size=1,
       rate=1,
       padding="SAME",
       dropout_rate=0,
       use_bias=True,
       activation_fn=None,
       training=True,
       scope="hc",
       reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    _inputs = inputs

    if padding.lower() == "causal":
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "valid"

    if filters is None:
        filters = inputs.get_shape().as_list()[-1]


    params = {"filters": 2*filters, "kernel_size": size,
              "dilation_rate": rate, "padding": padding, "use_bias": use_bias,
              "kernel_initializer": "he_normal"}

    tensor = tf.keras.layers.Conv1D(**params)(inputs)
    H1, H2 = tf.split(tensor, 2, axis=-1)
    H1 = normalize(H1, scope="H1")
    H2 = normalize(H2, scope="H2")
    H1 = tf.nn.sigmoid(H1, "gate")
    H2 = activation_fn(H2, "info") if activation_fn is not None else H2
    tensor = H1*H2 + (1.-H1)*_inputs

    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor

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
    n_iter = 50  # Number of inversion iterations
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


def conv1d_transpose(inputs,
                     filters=None,
                     size=3,
                     stride=2,
                     padding='same',
                     dropout_rate=0,
                     use_bias=True,
                     activation=None,
                     training=True,
                     scope="conv1d_transpose",
                     reuse=None):
    '''
        Args:
          inputs: A 3-D tensor with shape of [batch, time, depth].
          filters: An int. Number of outputs (=activation maps)
          size: An int. Filter size.
          rate: An int. Dilation rate.
          padding: Either `same` or `valid` or `causal` (case-insensitive).
          dropout_rate: A float of [0, 1].
          use_bias: A boolean.
          activation_fn: A string.
          training: A boolean. If True, dropout is applied.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
        Returns:
          A tensor of the shape with [batch, time*2, depth].
        '''
    if filters is None:
        filters = inputs.get_shape().as_list()[-1]
    inputs = tf.expand_dims(inputs, 1)
    tensor = tf.keras.layers.Conv2DTranspose(filters=filters,
                               kernel_size=(1, size),
                               strides=(1, stride),
                               padding=padding,
                               activation=None,
                               kernel_initializer="he_normal",
                               use_bias=use_bias)(inputs)
    tensor = tf.squeeze(tensor, 1)
    tensor = normalize(tensor)
    if activation is not None:
        tensor = activation(tensor)
    tensor = tf.keras.layers.Dropout(rate=dropout_rate)(tensor)

    return tensor

def normalize(inputs,
              scope="normalize",
              reuse=None):
    '''Applies layer normalization that normalizes along the last axis.
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over the last dimension.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    outputs = tf.keras.layers.LayerNormalization(axis=-1)(inputs)
    return outputs


def SSRN(training=True):
    ''' Spectrogram Super-resolution Network
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)
    Returns:
      Z: Spectrogram Predictions. (B, T, 1+n_fft/2)
    '''

    i = 1 # number of layers
    inputs = keras.layers.Input((128,hp.n_mels))

    # -> (B, T/r, c)
    tensor = conv1d(inputs, filters=hp.c, size=1, rate=1, dropout_rate=hp.dropout_rate, training=training, scope="C_{}".format(i))
    i += 1
    for j in range(2):
        tensor = hc(tensor, size=3, rate=3**j, dropout_rate=hp.dropout_rate, training=training, scope="HC_{}".format(i))
        i += 1
    for _ in range(2):
        # -> (B, T/2, c) -> (B, T, c)
        tensor = conv1d_transpose(tensor, scope="D_{}".format(i), dropout_rate=hp.dropout_rate, training=training,)
        i += 1
        for j in range(2):
            tensor = hc(tensor, size=3, rate=3**j, dropout_rate=hp.dropout_rate, training=training, scope="HC_{}".format(i))
            i += 1
    # -> (B, T, 2*c)
    tensor = conv1d(tensor, filters=2*hp.c, size=1, rate=1, dropout_rate=hp.dropout_rate, training=training, scope="C_{}".format(i))
    i += 1
    for _ in range(2):
        tensor = hc(tensor, size=3, rate=1, dropout_rate=hp.dropout_rate, training=training, scope="HC_{}".format(i))
        i += 1
    # -> (B, T, 1+n_fft/2)
    tensor = conv1d(tensor, filters=1+hp.n_fft//2, size=1, rate=1, dropout_rate=hp.dropout_rate, training=training, scope="C_{}".format(i))
    i += 1

    for _ in range(2):
        tensor = conv1d(tensor, size=1, rate=1, dropout_rate=hp.dropout_rate, activation_fn=tf.nn.relu, training=training, scope="C_{}".format(i))
        i += 1
    logits = conv1d(tensor, size=1, rate=1, dropout_rate=hp.dropout_rate, training=training, scope="C_{}".format(i))
    Z = tf.nn.sigmoid(logits)
    
    model = keras.models.Model(inputs, [logits,Z])
    return logits, Z, model 

def loss_bd_and_mels(y_true,y_pred):
    loss_mels = tf.reduce_mean(tf.abs(y_true-y_pred[1]))
    loss_bd1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[0], labels=y_true))

    return loss_mels+lossbd1



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
    y, _ = librosa.effects.trim(y)

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

def make_wave(freq, duration, sample_rate = 22050):
    wave = [i/((sample_rate/(2*np.pi))/freq) for i in range(0, int(duration))]
    wave = np.stack(wave)
    wave = np.cos(wave)
    '''
    sd.play(wave,sample_rate)
    cont = input("...")
    '''
    return wave


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
    


path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/synced/waveforms with gradient graphs/1"

test = load_wave(path+"/wavs")
mel,mag = load_spectrograms(test)
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(mel[:80])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(mag[:1024])
plt.show()

print(mag.shape,mel.shape)
print(spectrogram2wav(mag[:128*hp.r]).shape)
model = SSRN()[-1]
pred = model.predict(np.zeros((1,128,hp.n_mels)))[1][0]
print(pred.shape)
print(spectrogram2wav(pred).shape)
print("?")

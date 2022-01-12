import numpy as np 
from pyrir import Omni, Dipole, Cardioid, Subcardioid, Hypercardioid, Field, RIR, ReflectRoom, ReverbRoom
import sounddevice as sd
import librosa
import audiolab
import scipy

# Acoustic Field
fs = 22050 # sampling rate
n_sample = 1024 # number of supports of RIR train
field = Field(fs, n_sample=n_sample)

# Construct Room
rt60 = 10 # second
room = ReverbRoom((50,50,32), rt60)

# Microphon Array
azimuth_degree = 0
elevation_degree = 0
dipole = Dipole((2,1.5,1.6), (azimuth_degree,elevation_degree))
omni = Omni((2,1.5,1.6))

# speaker
doa = 0    # degree 
radius = 1.5 # meter
speaker = dipole.generate_speaker(radius, doa)

# setup speaker and mic array
room.setup_mic_speaker([dipole, omni], speaker)

rir_tuple = field.compute_rir(room)


# Reverb numpy Array List Supporting Multichannel Clean Audio (WAV format only for now) 
wave,_ = librosa.load("C:/ProgramData/Virtuoso/metadata/rendered.wav")
reverb_numpy_audio_list = rir_tuple[0].apply2audio1D(wave)

print(np.stack(reverb_numpy_audio_list).shape)
c = scipy.vstack((reverb_numpy_audio_list[0],reverb_numpy_audio_list[1]))
audiolab.wavwrite(c, "C:/ProgramData/Virtuoso/metadata/renderedflip.wav", fs, enc)

# Reverb audio folder
#speaker_audio_folder = 'speaker_audio_folder'
#rir_tuple[0].apply2audio_folder(speaker_audio_folder)

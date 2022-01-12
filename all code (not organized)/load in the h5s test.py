import h5py

midi_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning/Train Midis 0.h5"

h5f = h5py.File(midi_path,'r')
midis = h5f['all_data'][:]
h5f.close()

print(midis.shape)

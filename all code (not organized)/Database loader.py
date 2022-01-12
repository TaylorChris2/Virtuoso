import sounddevice as sd
import librosa
import midi
import skimage.transform
import numpy as np
import os
import h5py
import time
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
        track_array.append([[0,0,0,0],[0,0,0,0]])##one four channel list for pitch and one for articulation
    track_array = np.stack(track_array)
    
    ##filling in the track array with real note data
    for track in only_notes:
        current_tick = 0
        for n,event in enumerate(track):
            current_tick += event.tick
            if event.data[1] == 100:##every note start
                if event.data[0] < 50:
                    print(event.data)
                for i in range(0,4):
                    track_array[current_tick][1][i] = 1
                for i in range(current_tick,current_tick+track[n+1].tick):
                    slot = 0
                    while True:
                        if track_array[i][0][slot] == 0:
                            track_array[i][0][slot] = event.data[0]
                            break
                        slot += 1
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

set_size = 32
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data"
save_folder_path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/Midis and Mels for Machine Learning"
midis = []
mels = []
sets = 0
for set_ in os.listdir(path):
    print(set_)
    found_wav = False
    found_mid = False
    for file in os.listdir(os.path.join(path,set_)):
        if file.endswith(".wav") and not found_wav:
            mel,_ = librosa.load(os.path.join(os.path.join(path,set_), file))
            found_wav = True
        elif file.endswith("mid") and not found_mid:
            midi_array = load_midi_violin(os.path.join(os.path.join(path,set_), file))
            found_mid = True

    if not found_wav or not found_mid:
        print("Data incomplete. Failed to load: " + os.path.join(path,set_))
    else:
        sets+=1
        mel = np.stack([[mel,mel,mel,mel],[mel,mel,mel,mel]])
        z_zoom = mel.shape[0]/midi_array.shape[0]
        y_zoom = mel.shape[1]/midi_array.shape[1]
        x_zoom = mel.shape[2]/midi_array.shape[2]

        rescaled_midi = skimage.transform.rescale(midi_array, (z_zoom, y_zoom, x_zoom))

        index = 0
        segments = []
        while True:
            start = index*220500
            end = (index+1)*220500
            if np.array([[x[start:end] for x in a] for a in mel]).shape[2] == 0:
                break
            segments.append(np.array([[x[0:220500] for x in a] for a in mel]))
            index += 1
        ##padding the ending
        padding_amt = 220500-segments[-1].shape[2]
        for i,note in enumerate(segments[-1]):
            for j,channel in enumerate(note):
                segments[-1][i][j] = np.append(segments[-1][i][j], np.zeros(padding_amt))
        for segment in segments:
            x_zoom = 262144/segment.shape[2]
            mels.append(skimage.transform.rescale(segment, (1, 1, x_zoom)))
        
        index = 0
        segments = []
        while True:
            start = index*220500
            end = (index+1)*220500
            if np.array([[x[start:end] for x in a] for a in rescaled_midi]).shape[2] == 0:
                break
            segments.append(np.array([[x[0:220500] for x in a] for a in rescaled_midi]))
            index += 1
        ##padding the ending
        padding_amt = 220500-segments[-1].shape[2]
        for i,note in enumerate(segments[-1]):
            for j,channel in enumerate(note):
                segments[-1][i][j] = np.append(segments[-1][i][j], np.zeros(padding_amt))
        for segment in segments:
            x_zoom = 262144/segment.shape[2]
            midis.append(skimage.transform.rescale(segment, (1, 1, x_zoom)))
            
    ##testing by playing a wav
    z_zoom = 1/mels[0].shape[0]
    y_zoom = 1/mels[0].shape[1]
    x_zoom = 220500/mels[0].shape[2]
    playable_wav = skimage.transform.rescale(mels[0], (z_zoom, y_zoom, x_zoom))
    sd.play(np.squeeze(np.squeeze(playable_wav)), 22050)
    print(np.squeeze(np.squeeze(playable_wav)).shape)

print("Loaded in" ,len(midis),len(mels), "sets from", sets, "folders in", int((time.time() - start_time)/60), "minutes and",
          int(((time.time() - start_time) % 60)+1), "seconds.")
midi_sets, mel_sets = seperate_sets(midis, mels, set_size)

start_time = time.time()


print("\nSaving loaded data in: " + save_folder_path + "...")

if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)

for n, set_ in enumerate(midi_sets):
    train_midis, val_midis, test_midis = split_train_val_test(set_)
    
    save_data_set(train_midis, save_folder_path, "Train Midis "+str(n))
    save_data_set(val_midis, save_folder_path, "Val Midis "+str(n))
    save_data_set(test_midis, save_folder_path, "Test Midis "+str(n))

print("Finished saving images. Proceeding to save masks...")

for n, set_ in enumerate(mel_sets):
    train_mels, val_mels, test_mels = split_train_val_test(set_)
    
    save_data_set(train_mels, save_folder_path, "Train Mels "+str(n))
    save_data_set(val_mels, save_folder_path, "Val Mels "+str(n))
    save_data_set(test_mels, save_folder_path, "Test Mels "+str(n))

print("Finished saving masks.")
print("\nAll data finished saving in", int((time.time() - start_time)/60), "minutes and ",
    int(((time.time() - start_time) % 60)+1), "seconds.")


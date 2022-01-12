import sounddevice as sd
import librosa
import midi
import skimage.transform
import numpy as np



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
        
                
              
    

y,_ = librosa.load("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data/S3 Alg 1_4/A1-0001_allegro assai 1 of 4_00086400.wav")

midi_array = load_midi_violin("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/cut data/S3 Alg 1_4/allegro assai 1 of 4.mid")
y = np.stack([[y,y,y,y],[y,y,y,y]])
z_zoom = y.shape[0]/midi_array.shape[0]
y_zoom = y.shape[1]/midi_array.shape[1]
x_zoom = y.shape[2]/midi_array.shape[2]

rescaled_midi = skimage.transform.rescale(midi_array, (z_zoom, y_zoom, x_zoom))

np.array([[x[0:220500] for x in a] for a in y])

index = 0
segments = []
while True:
    start = index*220500
    end = (index+1)*220500
    if np.array([[x[start:end] for x in a] for a in y]).shape[2] == 0:
        break
    segments.append(np.array([[x[0:220500] for x in a] for a in y]))
    index += 1
del segments[-1]

print(len(segments))
sd.play(segments[0], 22050)

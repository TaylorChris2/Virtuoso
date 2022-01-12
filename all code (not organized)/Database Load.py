import sounddevice as sd
import librosa
import midi

#y,sr = librosa.load("C:/Users/JiangQin/Documents/python/Music Composition Project/code files/Text to speech testing/LJSpeech-1.1/wavs/LJ001-0001.wav")
#sd.play(y, 22050)


def load_midi(path):
    note_events = []
    mid = midi.read_midifile(path)
    for n,track in enumerate(mid):
        note_events.append([])
        for event in track:
            if "NoteOnEvent" in str(event):
                note_events[n].append(event)
    only_notes = []
    for n,track in enumerate(note_events):
        if len(track)>0:
            only_notes.append(track)
            print(track)
        else:
            print(track)


    track_lengths = []
    for n,track in enumerate(only_notes):
        track_lengths.append(0)
        for event in track:
            track_lengths[n] += event.tick
    track_length = max(track_lengths)
    track_array = []

    

    for i in range(0,track_length):
        channeled = []
        for channel in only_notes:
            channeled.append(0) 

        channeled.append(0) ##for the articulation
            
        track_array.append(channeled)
    
    
        
load_midi("C:/Users/JiangQin/Documents/python/Music Composition Project/Music Files/Violin/bach sonata 3/midi files/vs3-4alg.mid")

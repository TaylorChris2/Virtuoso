import midi
import librosa

mid = midi.read_midifile("C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/bach partita 1/midi files/vp1-3co.mid")

on = midi.NoteOnEvent(tick=0, velocity=20, pitch=midi.G_1)

print(on)

##getting only the note data

for track in mid:
    for event in track:
        if "NoteOnEvent" in str(event):
            note_events.append(event)

track_array

     

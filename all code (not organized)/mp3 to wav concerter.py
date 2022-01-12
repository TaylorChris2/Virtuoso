import sounddevice as sd
import librosa
from pydub import AudioSegment
import os

# files                                                                         
src = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music Files/Violin"
dst = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music Files/Violin/wavs"

for path, dirs, files in os.walk(src, topdown=False):
    for file in files:
        if file.endswith(".mp3"):
            print(os.path.join(path, file))
            # convert wav to mp3                                                            
            sound = AudioSegment.from_mp3(os.path.join(path, file))
            if not os.path.exists(dst):
                os.make_dirs(dst)
            sound.export(os.path.join(path, file.replace(".mp3",".wav")), format="wav")

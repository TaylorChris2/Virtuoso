from os import path
from pydub import AudioSegment
import os

# files                                                                         
path = "C:/Users/JiangQin/Documents/python/Music Composition Project/Music data/violin/all data"

for set_ in os.listdir(path):
    print("")
    for file in os.listdir(path+"/"+set_):
        print(file)
        if file.endswith("mid"):
            os.rename(path+"/"+set_+"/"+file,path+"/"+set_+"/midi.mid")
        if file.endswith("wav"):
            sound = AudioSegment.from_wav(path+"/"+set_+"/"+file)
            if "instrument.wav" not in file:
                os.remove(path+"/"+set_+"/"+file)
            #sound.export(path+"/"+set_+"/instrument.wav", format="wav")
      

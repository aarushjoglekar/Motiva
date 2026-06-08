from music.song import Song
import os

def song_midi_representation(song: Song):
    mid = song.to_midi()
    DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(DIR, "song_midi_representation")
    os.makedirs(filename, exist_ok=True)
    mid.save(filename=f"{filename}/{song.name}.mid")
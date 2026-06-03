import os
import numpy as np
from music import constants

class Song:
    RESOLUTION = 20 # per second

    TEMPLATE = [{
        "active_notes": np.zeros(88),
        "active_fingers": np.zeros(10)
    }]

    CHOPIN_WALTZ_OP69_NO1 = "chopin_waltz_op69_no1"

    def __init__(self, song_data: list):
        self.song_data = song_data

    def sample_at(self, time: float): #TODO LOOKAHEAD
        index = Song.time_to_index(time)
        done = index + 1 == len(self.song_data)
        return (self.song_data[index]["active_notes"], self.song_data[index]["active_fingers"]), done
    
    def total_time(self):
        return len(self.song_data) / Song.RESOLUTION

    @staticmethod
    def from_txt(name: str):
        DIR = os.path.dirname(os.path.abspath(__file__))
        with open(f"{DIR}/songs/{name}/{name}.txt") as file:
            song_data = []

            first_line = True
            for line in file:

                if first_line:
                    first_line = False
                    continue

                line = line.strip().split("\t")

                start_time = line[1]
                end_time = line[2]

                start_time_index = Song.time_to_index(float(start_time))
                end_time_index = Song.time_to_index(float(end_time))

                raw_note = line[3]
                note_value = constants.NOTES[raw_note[:-1]]
                octave = int(raw_note[-1])
                active_note = 8 * octave + note_value

                raw_finger = line[7]

                if "_" in raw_finger:
                    raw_finger = raw_finger[2]

                active_finger = constants.FINGER[int(raw_finger)]

                diff = end_time_index - len(song_data)
                if diff > 0:
                    song_data += Song.TEMPLATE * diff

                for index in range(start_time_index, end_time_index):
                    song_data[index]["active_notes"][active_note] = 1
                    song_data[index]["active_fingers"][active_finger] = 1

        return Song(song_data)
    
    @staticmethod
    def time_to_index(time: float):
        return round(time * Song.RESOLUTION)
import os
import numpy as np
from music import constants

class Song:
    RESOLUTION = 20 # per second
    LOOKAHEAD = 10

    NUM_PIANO_NOTES = 88
    NUM_ACTIVE_FINGERS = 10
    NUM_FEATURES = NUM_PIANO_NOTES + NUM_ACTIVE_FINGERS

    CHOPIN_WALTZ_OP69_NO1 = "chopin_waltz_op69_no1"

    def __init__(self, data: np.ndarray, fingers_to_keys_data: np.ndarray):
        self.data = data
        self.fingers_to_keys_data = fingers_to_keys_data
        self.length = len(self.data)

    def sample_at(self, time: float):
        index = Song.time_to_index(time)
        end = index + Song.LOOKAHEAD
        done = (self.length - index - 1) == 0

        if end <= self.length:
            samples = self.data[index:end]
        else:
            samples = np.pad(self.data[index:], ((0, end - self.length), (0, 0)), mode='constant')

        fingers_to_keys_sample = self.fingers_to_keys_data[index]

        return samples.ravel(), fingers_to_keys_sample, done
    
    def total_time(self):
        return self.length / Song.RESOLUTION

    @staticmethod
    def from_txt(name: str):
        DIR = os.path.dirname(os.path.abspath(__file__))
        with open(f"{DIR}/songs/{name}/{name}.txt") as file:
            data = []
            fingers_to_keys_data = []

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

                diff = end_time_index - len(data)
                if diff > 0:
                    data += [np.zeros(Song.NUM_FEATURES) for _ in range(diff)]
                    fingers_to_keys_data += [(np.zeros(Song.NUM_ACTIVE_FINGERS) - 1) for _ in range(diff)]

                for index in range(start_time_index, end_time_index):
                    data[index][active_note] = 1
                    data[index][active_finger + Song.NUM_PIANO_NOTES] = 1
                    fingers_to_keys_data[index][active_finger] = active_note

        return Song(np.array(data, dtype=int), np.array(fingers_to_keys_data, dtype=int))
    
    @staticmethod
    def time_to_index(time: float):
        return round(time * Song.RESOLUTION)
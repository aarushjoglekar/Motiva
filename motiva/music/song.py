import os
import numpy as np
from music import constants
import mido


class Song:
    RESOLUTION = 20  # per second
    LOOKAHEAD = 10

    NUM_PIANO_NOTES = 88
    NUM_ACTIVE_FINGERS = 10
    NUM_FEATURES = NUM_PIANO_NOTES + NUM_ACTIVE_FINGERS

    TWINKLE_TWINKLE_LITTLE_STAR = "twinkle_twinkle_little_star"
    SOMEWHERE_OVER_THE_RAINBOW = "somewhere_over_the_rainbow"
    CHOPIN_WALTZ_OP69_NO1 = "chopin_waltz_op69_no1"

    def __init__(self, name: str, data: np.ndarray, fingers_to_keys_data: np.ndarray):
        self.name = name
        self.data = data
        self.fingers_to_keys_data = fingers_to_keys_data
        self.length = len(self.data)

    def sample_at(self, time: float):
        index = min(Song.time_to_index(time), self.length - 1)
        end = index + Song.LOOKAHEAD
        truncated = index >= self.length - 1

        if end <= self.length:
            samples = self.data[index:end]
        else:
            samples = np.pad(
                self.data[index:], ((0, end - self.length), (0, 0)), mode="constant"
            )

        fingers_to_keys_sample = self.fingers_to_keys_data[index]

        return samples.ravel(), fingers_to_keys_sample, truncated

    def total_time(self):
        return self.length / Song.RESOLUTION
    
    def compare_to(self, ground_truth:"Song"):
        length = max(self.length, ground_truth.length)

        truth = np.zeros((length, Song.NUM_PIANO_NOTES), dtype=bool)
        truth[:ground_truth.length] = ground_truth.data[:, :Song.NUM_PIANO_NOTES].astype(bool)

        pred = np.zeros((length, Song.NUM_PIANO_NOTES), dtype=bool)
        pred[:self.length] = self.data[:, :Song.NUM_PIANO_NOTES].astype(bool)

        tp = np.logical_and(pred, truth).sum()
        fp = np.logical_and(pred, ~truth).sum()
        fn = np.logical_and(~pred, truth).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def to_midi(self):
        ticks_per_beat = 480
        seconds_per_tick = 60 / (120 * ticks_per_beat)
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        track = mid.add_track()

        def time_to_ticks(seconds):
            return round(seconds / seconds_per_tick)

        prev_notes = np.zeros(Song.NUM_PIANO_NOTES, dtype=bool)
        last_event_time = 0.0

        for i in range(self.length):
            current_notes = self.data[i, : Song.NUM_PIANO_NOTES].astype(bool)
            frame_time = i / Song.RESOLUTION

            note_ons = np.where(current_notes > prev_notes)[0]
            note_offs = np.where(current_notes < prev_notes)[0]

            for note in note_offs:
                delta = time_to_ticks(frame_time - last_event_time)
                track.append(
                    mido.Message("note_off", note=21 + note, velocity=0, time=delta)
                )
                last_event_time = frame_time

            for note in note_ons:
                delta = time_to_ticks(frame_time - last_event_time)
                track.append(
                    mido.Message("note_on", note=21 + note, velocity=64, time=delta)
                )
                last_event_time = frame_time

            prev_notes = current_notes

        final_time = self.length / Song.RESOLUTION
        for note in np.where(prev_notes)[0]:
            delta = time_to_ticks(final_time - last_event_time)
            track.append(
                mido.Message("note_off", note=21 + note, velocity=0, time=delta)
            )
            last_event_time = final_time

        return mid

    @staticmethod
    def time_to_index(time: float):
        return round(time * Song.RESOLUTION)

    @staticmethod
    def from_txt(name: str):
        DIR = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(DIR, f"songs/{name}/{name}.txt")) as file:
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
                active_note = 12 * (octave - 1) + note_value + 3

                raw_finger = line[7]

                if "_" in raw_finger:
                    raw_finger = raw_finger[2]

                active_finger = constants.FINGER[int(raw_finger)]

                diff = end_time_index - len(data)
                if diff > 0:
                    data += [np.zeros(Song.NUM_FEATURES) for _ in range(diff)]
                    fingers_to_keys_data += [
                        (np.zeros(Song.NUM_ACTIVE_FINGERS) - 1) for _ in range(diff)
                    ]

                for index in range(start_time_index, end_time_index - 1):
                    data[index][active_note] = 1
                    data[index][active_finger + Song.NUM_PIANO_NOTES] = 1
                    fingers_to_keys_data[index][active_finger] = active_note

        return Song(
            name, np.array(data, dtype=int), np.array(fingers_to_keys_data, dtype=int)
        )
    
    @staticmethod
    def from_midi_string(name: str): # finger data left empty
        DIR = os.path.dirname(os.path.abspath(__file__))
        midi = mido.MidiFile(os.path.join(DIR, f"songs/{name}/{name}.mid"))

        return Song.from_midi(name=name, midi=midi)   

    @staticmethod
    def from_midi(name: str, midi: mido.MidiFile):
        notes = []
        active_notes = {}
        abs_time = 0.0

        for msg in midi:
            abs_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = abs_time

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                onset = active_notes.pop(msg.note, None)
                if onset is None:
                    continue
                note_index = msg.note - 21
                if 0 <= note_index < Song.NUM_PIANO_NOTES:
                    notes.append(
                        (
                            note_index,
                            Song.time_to_index(onset),
                            Song.time_to_index(abs_time),
                        )
                    )

        length = max((end for _, _, end in notes), default=0)
        data = np.zeros((length, Song.NUM_FEATURES), dtype=int)
        fingers_to_keys_data = (
            np.zeros((length, Song.NUM_ACTIVE_FINGERS), dtype=int) - 1
        )

        for note_index, start, end in notes:
            data[max(0, start) : max(0, end), note_index] = 1

        return Song(name, data, fingers_to_keys_data)     
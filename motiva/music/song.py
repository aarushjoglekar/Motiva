import os
import numpy as np
from music import constants
import mido


class SongSelection:
    DEBUG = "debug"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type


class Song:
    RESOLUTION = 20  # per second
    LOOKAHEAD = 10
    START_BUFFER = 10

    NUM_PIANO_NOTES = 88
    NUM_FINGERS = 10
    NUM_FEATURES = NUM_PIANO_NOTES + NUM_FINGERS

    TWINKLE_TWINKLE_LITTLE_STAR = SongSelection(
        "twinkle_twinkle_little_star", SongSelection.DEBUG
    )
    SOMEWHERE_OVER_THE_RAINBOW = SongSelection(
        "somewhere_over_the_rainbow", SongSelection.DEBUG
    )
    ANOTHER_LOVE = SongSelection("another_love", SongSelection.DEBUG)

    SOMEONE_LIKE_YOU_ADELE = SongSelection(
        "someone_like_you_adele", SongSelection.LEVEL_1
    )
    BELIEVER_IMAGINE_DRAGONS = SongSelection(
        "believer_imagine_dragons", SongSelection.LEVEL_1
    )
    PAYPHONE_WIZ_KHALIFA = SongSelection("payphone_wiz_khalifa", SongSelection.LEVEL_1)

    def __init__(
        self,
        name: str,
        type: str,
        data: np.ndarray,
        fingers_to_keys_data: np.ndarray,
        onset_velocity_data: np.ndarray,
        offset_velocity_data: np.ndarray,
    ):
        self.name = name
        self.type = type
        self.data = data
        self.fingers_to_keys_data = fingers_to_keys_data
        self.onset_velocity_data = onset_velocity_data
        self.offset_velocity_data = offset_velocity_data
        self.length = len(self.data)

    def sample_at(
        self,
        time: float,
        include_fingering_data: bool,
        include_onset_velocity_data: bool,
    ):
        index = min(Song.time_to_index(time), self.length - 1)
        end = index + Song.LOOKAHEAD
        truncated = index >= self.length - 1

        if end <= self.length:
            samples = self.data[index:end]
        else:
            samples = np.pad(
                self.data[index:], ((0, end - self.length), (0, 0)), mode="constant"
            )

        active_fingers = samples[0, Song.NUM_PIANO_NOTES : Song.NUM_FEATURES]

        if not include_fingering_data:
            samples = samples[:, : Song.NUM_PIANO_NOTES]

        if include_onset_velocity_data:
            if end <= self.length:
                onset_samples = self.onset_velocity_data[index:end]
            else:
                onset_samples = np.pad(
                    self.onset_velocity_data[index:],
                    ((0, end - self.length), (0, 0)),
                    mode="constant",
                )
            samples = np.concatenate([samples, onset_samples], axis=1)

        fingers_to_keys_sample = self.fingers_to_keys_data[index]

        return samples.ravel(), fingers_to_keys_sample, active_fingers, truncated

    def total_time(self):
        return self.length / Song.RESOLUTION

    def compare_to(self, ground_truth: "Song"):
        length = max(self.length, ground_truth.length)

        truth = np.zeros((length, Song.NUM_PIANO_NOTES), dtype=bool)
        truth[: ground_truth.length] = ground_truth.data[
            :, : Song.NUM_PIANO_NOTES
        ].astype(bool)

        pred = np.zeros((length, Song.NUM_PIANO_NOTES), dtype=bool)
        pred[: self.length] = self.data[:, : Song.NUM_PIANO_NOTES].astype(bool)

        tp = np.logical_and(pred, truth).sum()
        fp = np.logical_and(pred, ~truth).sum()
        fn = np.logical_and(~pred, truth).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

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
                offset_velocity = int(round(self.offset_velocity_data[i, note] * 127))
                track.append(
                    mido.Message(
                        "note_off", note=21 + note, velocity=offset_velocity, time=delta
                    )
                )
                last_event_time = frame_time

            for note in note_ons:
                delta = time_to_ticks(frame_time - last_event_time)
                onset_velocity = max(
                    1, min(127, int(round(self.onset_velocity_data[i, note] * 127)))
                )
                track.append(
                    mido.Message(
                        "note_on", note=21 + note, velocity=onset_velocity, time=delta
                    )
                )
                last_event_time = frame_time

            prev_notes = current_notes

        final_time = self.length / Song.RESOLUTION
        for note in np.where(prev_notes)[0]:
            delta = time_to_ticks(final_time - last_event_time)
            offset_velocity = int(
                round(self.offset_velocity_data[self.length - 1, note] * 127)
            )
            track.append(
                mido.Message(
                    "note_off", note=21 + note, velocity=offset_velocity, time=delta
                )
            )
            last_event_time = final_time

        return mid

    @staticmethod
    def time_to_index(time: float):
        return round(time * Song.RESOLUTION)

    @staticmethod
    def from_txt(song: SongSelection):
        DIR = os.path.dirname(os.path.abspath(__file__))
        with open(
            os.path.join(DIR, f"songs/{song.type}/{song.name}/{song.name}.txt")
        ) as file:
            data = []
            fingers_to_keys_data = []
            onset_velocity_data = []
            offset_velocity_data = []

            for line in file:
                line = line.strip().split("\t")

                start_time = line[1]
                end_time = line[2]

                start_time_index = Song.time_to_index(float(start_time))
                end_time_index = Song.time_to_index(float(end_time))

                raw_note = line[3]
                note_value = constants.NOTES[raw_note[:-1]]
                octave = int(raw_note[-1])
                active_note = 12 * (octave - 1) + note_value + 3

                onset_velocity = float(line[4]) / 127.0
                offset_velocity = float(line[5]) / 127.0

                raw_finger = line[7]
                if "_" in raw_finger:
                    raw_finger = raw_finger[2]
                active_finger = constants.FINGER[int(raw_finger)]

                diff = end_time_index - len(data)
                if diff > 0:
                    data += [
                        np.zeros(Song.NUM_FEATURES, dtype=int) for _ in range(diff)
                    ]
                    fingers_to_keys_data += [
                        (np.zeros(Song.NUM_FINGERS) - 1) for _ in range(diff)
                    ]
                    onset_velocity_data += [
                        np.zeros(Song.NUM_PIANO_NOTES, dtype=np.float32)
                        for _ in range(diff)
                    ]
                    offset_velocity_data += [
                        np.zeros(Song.NUM_PIANO_NOTES, dtype=np.float32)
                        for _ in range(diff)
                    ]

                for index in range(start_time_index, end_time_index - 1):
                    data[index][active_note] = 1
                    data[index][active_finger + Song.NUM_PIANO_NOTES] = 1
                    fingers_to_keys_data[index][active_finger] = active_note

                onset_velocity_data[start_time_index][active_note] = onset_velocity
                offset_velocity_data[end_time_index - 1][active_note] = offset_velocity

        data = np.array(data, dtype=int)
        data = np.concatenate(
            [np.zeros((Song.START_BUFFER, Song.NUM_FEATURES), dtype=int), data],
            axis=0,
        )

        fingers_to_keys_data = np.array(fingers_to_keys_data, dtype=int)
        fingers_to_keys_data = np.concatenate(
            [
                np.zeros((Song.START_BUFFER, Song.NUM_FINGERS), dtype=np.int16) - 1,
                fingers_to_keys_data,
            ],
            axis=0,
        )

        start_buffer_float = np.zeros(
            (Song.START_BUFFER, Song.NUM_PIANO_NOTES), dtype=np.float32
        )

        onset_velocity_data = np.array(onset_velocity_data, dtype=np.float32)
        onset_velocity_data = np.concatenate(
            [start_buffer_float, onset_velocity_data], axis=0
        )

        offset_velocity_data = np.array(offset_velocity_data, dtype=np.float32)
        offset_velocity_data = np.concatenate(
            [start_buffer_float, offset_velocity_data], axis=0
        )

        return Song(
            name=song.name,
            type=song.type,
            data=data,
            fingers_to_keys_data=fingers_to_keys_data,
            onset_velocity_data=onset_velocity_data,
            offset_velocity_data=offset_velocity_data,
        )

    @staticmethod
    def from_midi_file(
        song: SongSelection, should_add_start_buffer: bool = True
    ):  # no finger data
        DIR = os.path.dirname(os.path.abspath(__file__))
        midi = mido.MidiFile(
            os.path.join(DIR, f"songs/{song.type}/{song.name}/{song.name}.mid")
        )
        return Song.from_midi(
            name=song.name,
            type=song.type,
            should_add_start_buffer=should_add_start_buffer,
            midi=midi,
        )

    @staticmethod
    def from_midi(
        name: str, type: str, should_add_start_buffer: bool, midi: mido.MidiFile
    ):  # no finger data
        notes = []
        active_notes = {}
        abs_time = 0.0

        for msg in midi:
            abs_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = (abs_time, msg.velocity)

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                onset = active_notes.pop(msg.note, None)
                if onset is None:
                    continue
                onset_time, onset_velocity = onset
                note_index = msg.note - 21
                if 0 <= note_index < Song.NUM_PIANO_NOTES:
                    notes.append(
                        (
                            note_index,
                            Song.time_to_index(onset_time),
                            Song.time_to_index(abs_time),
                            onset_velocity,
                            msg.velocity,
                        )
                    )

        length = max((max(end, start + 1) for _, start, end, _, _ in notes), default=0)
        data = np.zeros((length, Song.NUM_FEATURES), dtype=int)
        fingers_to_keys_data = np.zeros((length, Song.NUM_FINGERS), dtype=int) - 1
        onset_velocity_data = np.zeros((length, Song.NUM_PIANO_NOTES), dtype=np.float32)
        offset_velocity_data = np.zeros(
            (length, Song.NUM_PIANO_NOTES), dtype=np.float32
        )

        for (
            note_index,
            start,
            end,
            onset_velocity,
            offset_velocity,
        ) in notes:
            start = max(0, start)
            end = max(start + 1, end)
            data[start:end, note_index] = 1
            onset_velocity_data[start, note_index] = onset_velocity / 127.0
            offset_velocity_data[end - 1, note_index] = offset_velocity / 127.0

        if should_add_start_buffer:
            data = np.concatenate(
                [np.zeros((Song.START_BUFFER, Song.NUM_FEATURES), dtype=int), data],
                axis=0,
            )
            fingers_to_keys_data = np.concatenate(
                [
                    np.zeros((Song.START_BUFFER, Song.NUM_FINGERS), dtype=int) - 1,
                    fingers_to_keys_data,
                ]
            )
            start_buffer_float = np.zeros(
                (Song.START_BUFFER, Song.NUM_PIANO_NOTES), dtype=np.float32
            )
            onset_velocity_data = np.concatenate(
                [start_buffer_float, onset_velocity_data], axis=0
            )
            offset_velocity_data = np.concatenate(
                [start_buffer_float, offset_velocity_data], axis=0
            )

        return Song(
            name=name,
            type=type,
            data=data,
            fingers_to_keys_data=fingers_to_keys_data,
            onset_velocity_data=onset_velocity_data,
            offset_velocity_data=offset_velocity_data,
        )

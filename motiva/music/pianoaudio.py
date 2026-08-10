import os
import fluidsynth
import numpy as np
import mido


class PianoAudio:
    NOTES_BEFORE_A1 = 21
    MAX_QVEL = 4.5
    MIN_AUDIO_VEL = 1
    MAX_AUDIO_VEL = 127
    GAMMA = 0.5

    def __init__(self, never_play_audio: bool, press_thresholds: np.ndarray):
        self.press_thresholds = press_thresholds

        if not never_play_audio:
            DIR = os.path.dirname(os.path.abspath(__file__))
            self.fluidsynth = fluidsynth.Synth()
            self.fluidsynth.setting("synth.gain", 2.0)
            self.fluidsynth.start()
            sfid = self.fluidsynth.sfload(os.path.join(DIR, "soundfonts/TimGM6mb.sf2"))
            self.fluidsynth.program_select(0, sfid, 0, 0)

    def reset(
        self, play_audio: bool, record_midi: bool, save_midi: bool, midi_file: str
    ):
        self.play_audio = play_audio
        self.record_midi = record_midi
        self.save_midi = save_midi
        self.midi_file = midi_file
        self.is_useless = (
            not self.play_audio and not self.record_midi and not self.save_midi
        )

        if self.record_midi or self.save_midi:
            self.last_event_time = 0
            self.ticks_per_beat = 480
            self.seconds_per_tick = 60 / (120 * self.ticks_per_beat)
            self.mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
            self.track = self.mid.add_track()

        self.key_pressed = np.zeros(88, dtype=bool)

    def update(
        self, piano_qpos: np.ndarray, piano_qvel: np.ndarray, episode_time: float
    ):
        currently_pressed = piano_qpos > self.press_thresholds
        new_onsets = currently_pressed & ~self.key_pressed
        new_offsets = ~currently_pressed & self.key_pressed

        if not self.is_useless:
            for i in np.where(new_onsets)[0]:
                velocity_norm = float(
                    PianoAudio.compute_velocity_norm(float(piano_qvel[i]))
                )
                listening_velocity = round(
                    PianoAudio.MIN_AUDIO_VEL
                    + (PianoAudio.MAX_AUDIO_VEL - PianoAudio.MIN_AUDIO_VEL)
                    * (velocity_norm**PianoAudio.GAMMA)
                )
                midi_velocity = round(
                    PianoAudio.MIN_AUDIO_VEL
                    + (PianoAudio.MAX_AUDIO_VEL - PianoAudio.MIN_AUDIO_VEL)
                    * velocity_norm
                )
                # print(f"QVel: {piano_qvel[i]} || Vel: {midi_velocity}")
                if self.play_audio:
                    self.fluidsynth.noteon(
                        0, PianoAudio.NOTES_BEFORE_A1 + i, listening_velocity
                    )
                if self.record_midi or self.save_midi:
                    self.track.append(
                        mido.Message(
                            "note_on",
                            note=PianoAudio.NOTES_BEFORE_A1 + i,
                            velocity=midi_velocity,
                            time=self.calculate_delta_ticks(episode_time),
                        )
                    )

            for i in np.where(new_offsets)[0]:
                if self.play_audio:
                    self.fluidsynth.noteoff(0, PianoAudio.NOTES_BEFORE_A1 + i)
                if self.record_midi or self.save_midi:
                    self.track.append(
                        mido.Message(
                            "note_off",
                            note=PianoAudio.NOTES_BEFORE_A1 + i,
                            velocity=0,
                            time=self.calculate_delta_ticks(episode_time),
                        )
                    )

        self.key_pressed = currently_pressed

        return new_onsets

    def calculate_delta_ticks(self, episode_time: float):
        delta_ticks = int((episode_time - self.last_event_time) / self.seconds_per_tick)
        self.last_event_time = episode_time
        return delta_ticks

    def save_and_close(self):
        if self.play_audio:
            self.fluidsynth.all_notes_off(0)

        if self.save_midi:
            self.mid.save(filename=self.midi_file)

        if self.record_midi:
            return self.mid

    @staticmethod
    def compute_velocity_norm(q_vel: float | np.ndarray):
        return np.clip(np.abs(q_vel) / PianoAudio.MAX_QVEL, 0.0, 1.0)

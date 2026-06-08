from datetime import datetime
import os
import fluidsynth
import numpy as np

class PianoAudio:
    PRESS_THRESHOLD = 0.15
    MAX_QVEL = 3

    def __init__(self, play_audio: bool, record_midi: bool, midi_path:str):
        self.play_audio = play_audio
        self.record_midi = record_midi
        self.midi_path = midi_path
        self.key_pressed = np.zeros(88, dtype=bool)

        if self.play_audio:
            DIR = os.path.dirname(os.path.abspath(__file__))
            self.fluidsynth = fluidsynth.Synth()
            self.fluidsynth.start()
            sfid = self.fluidsynth.sfload(os.path.join(DIR, "soundfonts/TimGM6mb.sf2"))
            self.fluidsynth.program_select(0, sfid, 0, 0)

        self.is_useless = not self.play_audio and not self.record_midi

    def update(self, piano_qpos: np.ndarray, piano_qvel: np.ndarray, episode_time: float):
        if self.is_useless:
            return

        for i in range(88):
            pressed = piano_qpos[i] > self.PRESS_THRESHOLD
            was_pressed = self.key_pressed[i]

            if pressed and not was_pressed:
                velocity = max(1, int(np.clip(abs(piano_qvel[i]) / self.MAX_QVEL, 0.0, 1.0) * 127)) # scale from 1 to 127 (0 is off so avoid that)
                if self.play_audio:
                    self.fluidsynth.noteon(0, 21 + i, velocity)
                # if self.record_midi:
                    # self.recorder.note_on(i, velocity, episode_time)

            elif not pressed and was_pressed:
                if self.play_audio:
                    self.fluidsynth.noteoff(0, 21 + i)
                # if self.record_midi:
                    # self.recorder.note_off(i, episode_time)

            self.key_pressed[i] = pressed

    def save(self):
        if self.record_midi:
            path = os.path.join(self.midi_path, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            pass # save the midi file
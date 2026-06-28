import os
import fluidsynth
import numpy as np
import mido

class PianoAudio:
    PRESS_THRESHOLD = 0.75
    MAX_QVEL = 3

    def __init__(self, play_audio: bool, record_midi: bool, save_midi: bool, midi_file:str):
        self.play_audio = play_audio
        self.record_midi = record_midi
        self.save_midi = save_midi
        self.midi_file = midi_file
        self.key_pressed = np.zeros(88, dtype=bool)

        if self.play_audio:
            DIR = os.path.dirname(os.path.abspath(__file__))
            self.fluidsynth = fluidsynth.Synth()
            self.fluidsynth.start()
            sfid = self.fluidsynth.sfload(os.path.join(DIR, "soundfonts/TimGM6mb.sf2"))
            self.fluidsynth.program_select(0, sfid, 0, 0)

        if self.record_midi or self.save_midi:
            self.last_event_time = 0
            self.ticks_per_beat = 480
            self.seconds_per_tick = 60 / (120 * self.ticks_per_beat)
            self.mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
            self.track = self.mid.add_track()

        self.is_useless = not self.play_audio and not self.record_midi and not self.save_midi

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
                if self.record_midi or self.save_midi:
                    self.track.append(mido.Message(
                        "note_on",
                        note=21 + i,
                        velocity=velocity,
                        time=self.calculate_delta_ticks(episode_time)
                    ))
            elif not pressed and was_pressed:
                if self.play_audio:
                    self.fluidsynth.noteoff(0, 21 + i)
                if self.record_midi or self.save_midi:
                    self.track.append(mido.Message(
                        "note_off",
                        note=21 + i,
                        velocity=0,
                        time=self.calculate_delta_ticks(episode_time)
                    ))

            self.key_pressed[i] = pressed

    def calculate_delta_ticks(self, episode_time: float):
        delta_ticks = int((episode_time - self.last_event_time) / self.seconds_per_tick)
        self.last_event_time = episode_time
        return delta_ticks

    def save_and_close(self):
        if self.play_audio:
            self.fluidsynth.all_notes_off(0)
            self.fluidsynth.delete()

        if self.save_midi:
            self.mid.save(filename=self.midi_file)

        if self.record_midi:
            return self.mid
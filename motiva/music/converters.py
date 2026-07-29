from music.song import Song
from music.constants import NOTE_NAMES_NO_OVERLAP
import os
import mido


def song_midi_representation(song: Song):
    mid = song.to_midi()
    DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(DIR, f"songs/song_midi_representation")
    os.makedirs(filename, exist_ok=True)
    mid.save(filename=f"{filename}/{song.name}.mid")


def midi_to_txt(input_path: str, output_path: str):
    mid = mido.MidiFile(input_path)
    ticks_per_beat = mid.ticks_per_beat

    notes = []
    note_id = 0

    merged = mido.merge_tracks(mid.tracks)

    tempo = 500_000
    current_tick = 0
    active_notes = {}

    for msg in merged:
        current_tick += msg.time

        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue

        is_note_on = msg.type == "note_on" and msg.velocity > 0
        is_note_off = msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        )

        if is_note_on:
            key = (msg.channel, msg.note)
            active_notes[key] = {
                "onset_tick": current_tick,
                "onset_velocity": msg.velocity,
                "onset_tempo": tempo,
            }

        elif is_note_off:
            key = (msg.channel, msg.note)
            if key not in active_notes:
                continue

            onset_info = active_notes.pop(key)
            onset_time = (onset_info["onset_tick"] * onset_info["onset_tempo"]) / (
                ticks_per_beat * 1_000_000
            )
            offset_time = (current_tick * tempo) / (ticks_per_beat * 1_000_000)

            octave = (msg.note // 12) - 1
            name = NOTE_NAMES_NO_OVERLAP[msg.note % 12]

            notes.append(
                {
                    "note_id": note_id,
                    "onset_time": onset_time,
                    "offset_time": offset_time,
                    "spelled_pitch": f"{name}{octave}",
                    "onset_velocity": onset_info["onset_velocity"],
                    "offset_velocity": msg.velocity,
                    "channel": msg.channel,
                }
            )
            note_id += 1

    notes.sort(key=lambda n: (n["onset_time"], n["channel"], n["spelled_pitch"]))

    for i, note in enumerate(notes):
        note["note_id"] = i

    with open(output_path, "w") as f:
        for note in notes:
            line = "\t".join(
                [
                    str(note["note_id"]),
                    f"{note['onset_time']:.6f}",
                    f"{note['offset_time']:.6f}",
                    note["spelled_pitch"],
                    str(note["onset_velocity"]),
                    str(note["offset_velocity"]),
                    str(note["channel"]),
                    "FINGER_NUMBER",
                ]
            )
            f.write(line + "\n")

import numpy as np

def make_rescaler(old_min, old_max, new_min, new_max):
    diff = old_max - old_min
    scale = np.where(diff == 0, 1, (new_max - new_min) / np.where(diff == 0, 1, diff))
    offset = np.where(diff == 0, 0, new_min - old_min * scale)
    return scale, offset

def rescale(val: float | np.ndarray, scale: float | np.ndarray, offset: float | np.ndarray):
    return val * scale + offset
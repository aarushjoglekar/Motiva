import numpy as np

# prepares scale and offset for rescale
def make_rescaler(old_min, old_max, new_min, new_max):
    diff = old_max - old_min
    scale = np.where(diff == 0, 1, (new_max - new_min) / np.where(diff == 0, 1, diff))
    offset = np.where(diff == 0, 0, new_min - old_min * scale)
    return scale, offset

# linearly rescales with scale and offset from make_rescaler
def rescale(val: float | np.ndarray, scale: float | np.ndarray, offset: float | np.ndarray):
    return val * scale + offset

# scales reward with a gaussian
def proximity_reward(x: float | np.ndarray, lower: float, upper: float, margin: float, value_at_margin: float):
    scale = np.sqrt(-2 * np.log(value_at_margin))
    diff = np.where(x < lower, lower - x, np.where(x > upper, x - upper, 0.0)) / margin
    return np.where((lower <= x) & (x <= upper), 1.0, np.exp(-0.5 * (diff * scale) ** 2))
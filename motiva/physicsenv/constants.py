import numpy as np

JOINTS = [
    "forearm_tx",
    "forearm_ty",
    "forearm_tz",
    "forearm_roll",
    "WRJ2",
    "WRJ1",
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
]

# ordered to match 1-10 convention from pig dataset
FINGER_SITE = [
    "lh_lfdistal_tip",
    "lh_rfdistal_tip",
    "lh_mfdistal_tip",
    "lh_ffdistal_tip",
    "lh_thdistal_tip",
    "rh_thdistal_tip",
    "rh_ffdistal_tip",
    "rh_mfdistal_tip",
    "rh_rfdistal_tip",
    "rh_lfdistal_tip"
]

HANDS = [
    "rh",
    "lh"
]

ACTUATED_JOINT_NAMES = [
    "WRJ2",
    "WRJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
    "FFJ4",
    "FFJ3",
    "FFJ0",
    "MFJ4",
    "MFJ3",
    "MFJ0",
    "RFJ4",
    "RFJ3",
    "RFJ0",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ0",
    "forearm_tx",
    "forearm_ty",
    "forearm_tz",
    "forearm_roll",
]

ACTUATED_JOINT_KP_POSITION = np.array([
    10,
    8,
    0.4,
    1,
    0.5,
    1.5,
    1,
    1,
    1,
    0.5,
    1,
    1,
    0.5,
    1,
    1,
    0.5,
    1,
    1,
    1,
    0.5,
    100,
    300,
    300,
    300,
] * len(HANDS), dtype=np.float64)

ACTUATED_JOINT_DAMPING_POSITION = np.array([
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    45,
    45,
    40,
    0,
] * len(HANDS), dtype=np.float64)

# dof_frictionloss
ACTUATED_JOINT_KS_VELOCITY = np.array([
    0.01,
    0.01,
    0.01,
    0.01, 
    0.01, 
    0.01, 
    0.01, 
    0.01, 
    0.01,
    0.01, 
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.0,
    0.0,
    0.0,
    0.0,
] * len(HANDS), dtype=np.float64)

# joint: dof_damping
# tendon: 1 / sum(c_i^2 / damping_i)
ACTUATED_JOINT_KV_VELOCITY = np.array([
    0.5,
    0.5,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.05,
    0.025,
    0.05,
    0.05,
    0.025,
    0.05,
    0.05,
    0.025,
    0.05,
    0.05,
    0.05,
    0.025,
    5.0,
    5.0,
    5.0,
    2.0,
] * len(HANDS), dtype=np.float64)

ACTUATED_JOINT_KP_VELOCITY = np.array([
    0, # 0
    0, # 1
    0, # 2
    0, # 3
    0, # 4
    0, # 5
    0, # 6
    0, # 7
    0, # 8
    0, # 9
    0, # 10
    0, # 11
    0, # 12
    0, # 13
    0, # 14
    0, # 15
    0, # 16
    0, # 17
    0, # 18
    0, # 19
    70, # 20
    70, # 21
    70, # 22
    0, # 23
] * len(HANDS), dtype=np.float64)

ACTUATED_JOINT_MAX_SPEEDS = np.array([
    1,
    1.7,
    2,
    2,
    1,
    2,
    1.5,
    1,
    2,
    3,
    1,
    2, 
    3, 
    1, 
    2, 
    3, 
    2, 
    1, 
    2,
    3,
    0.04,
    0.3,
    0.075,
    0.7,
] * len(HANDS), dtype=np.float64)
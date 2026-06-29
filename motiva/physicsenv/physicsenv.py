import mujoco
import os
import numpy as np
from physicsenv import constants
from helpers import helpers
from music.song import Song


class PhysicsEnv:
    def __init__(self, seed: int):
        # instantiation
        self.model, self.data, piano_y_min, piano_y_max = self.initialize_models()
        self.viewer = None

        # ty joint ides
        self.ry_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_ty"
        )
        self.ly_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_ty"
        )

        # tz actuator and joint ids
        self.rz_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_forearm_tz"
        )
        self.lz_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lh_A_forearm_tz"
        )
        self.rz_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_tz"
        )
        self.lz_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_tz"
        )

        # range of actions
        self.action_lows = self.model.actuator_ctrlrange[:, 0]
        self.action_highs = self.model.actuator_ctrlrange[:, 1]

        # piano joint/site ids
        black_keys = {1, 4, 6, 9, 11}

        self.piano_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, key_name)
            for key_name in [
                f"{'black' if i % 12 in black_keys else 'white'}_joint_{i}"
                for i in range(88)
            ]
        ]

        self.piano_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                for site_name in [
                    f"{'black' if i % 12 in black_keys else 'white'}_key_site_{i}"
                    for i in range(88)
                ]
            ],
            dtype=int,
        )

        # piano joint rescaling
        self.piano_scale, self.piano_offset = helpers.make_rescaler(
            self.model.jnt_range[self.piano_joint_ids, 0],
            self.model.jnt_range[self.piano_joint_ids, 1],
            np.zeros(len(self.piano_joint_ids)) - 1,
            np.ones(len(self.piano_joint_ids)),
        )

        # hand joint ids
        self.hand_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in [
                f"{hand}_{joint}"
                for hand in constants.HANDS
                for joint in constants.JOINTS
            ]
        ]

        # hand joint rescaling
        self.hand_joint_scale, self.hand_joint_offset = helpers.make_rescaler(
            self.model.jnt_range[self.hand_joint_ids, 0],
            self.model.jnt_range[self.hand_joint_ids, 1],
            np.zeros(len(self.hand_joint_ids)) - 1,
            np.ones(len(self.hand_joint_ids)),
        )

        # finger site ids
        self.finger_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, finger_site)
                for finger_site in constants.FINGER_SITE
            ],
            dtype=int,
        )

        # finger site rescaling (values determined experimentally)
        finger_min = [-0.05, piano_y_min, 0]
        finger_max = [0.17, piano_y_max, 0.3]
        self.finger_site_scale, self.finger_site_offset = helpers.make_rescaler(
            np.array(finger_min * len(self.finger_site_ids)),
            np.array(finger_max * len(self.finger_site_ids)),
            np.zeros(len(finger_min) * len(self.finger_site_ids)) - 1,
            np.ones(len(finger_max) * len(self.finger_site_ids)),
        )

        # physics steps per env step
        self.physics_steps_per_env_step = round(
            (1 / Song.RESOLUTION) / self.model.opt.timestep
        )
        
        # random state
        self.rng = np.random.RandomState(seed)

    # action is a list indexed by actuator id of position values from -1 to 1
    # step will automatically scale based on each control range
    def step(self, action: np.ndarray):
        # scale actions
        scaled_action = self.action_lows + (action + 1) * 0.5 * (
            self.action_highs - self.action_lows
        )

        # pd position control
        self.data.ctrl[:] = scaled_action

        for _ in range(self.physics_steps_per_env_step):
            mujoco.mj_step(self.model, self.data)

        return self.get_obs()

    def get_obs(self):
        # qpos -> all joint positions (piano keys + each hand)
        # xpos -> forearm positions

        return (
            helpers.rescale(
                self.data.qpos[self.piano_joint_ids],
                self.piano_scale,
                self.piano_offset,
            ),
            helpers.rescale(
                self.data.qpos[self.hand_joint_ids],
                self.hand_joint_scale,
                self.hand_joint_offset,
            ),
            helpers.rescale(
                self.data.site_xpos[self.finger_site_ids].ravel(),
                self.finger_site_scale,
                self.finger_site_offset,
            ),
        )

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
        self.viewer.sync()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        offset = self.rng.uniform(-0.05, 0.05)
        for joint_id in [self.ry_joint_id, self.ly_joint_id]:
            lo = self.model.jnt_range[joint_id, 0]
            hi = self.model.jnt_range[joint_id, 1]
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = np.clip(
                self.data.qpos[self.model.jnt_qposadr[joint_id]] + offset, lo, hi
            )

        mujoco.mj_forward(self.model, self.data)

    def viewer_running(self):
        return self.viewer is None or self.viewer.is_running()

    def initialize_models(
        self, hover_offset: float = 0.12, forward_offset: float = 0.4
    ):
        DIR = os.path.dirname(os.path.abspath(__file__))
        model = mujoco.MjModel.from_xml_path(os.path.join(DIR, "models/world.xml"))
        data = mujoco.MjData(model)

        # apply gravity compensation to all hand bodies
        for id in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, id)
            if name is not None and (name.startswith("rh_") or name.startswith("lh_")):
                model.body_gravcomp[id] = 1

        # harden piano key contacts
        physics_timestep = model.opt.timestep

        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is not None and (
                "white_key_geom" in name or "black_key_geom" in name
            ):
                model.geom_solref[geom_id, 0] = (
                    physics_timestep * 2
                )  # must be >=2 for collision resolution to be stable
                model.geom_solref[geom_id, 1] = 1.0

        # ids of the first and last key
        first_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "white_key_0")
        last_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "white_key_87")

        # position and size of the piano
        y_min = model.body_pos[first_key_id][1]
        y_max = model.body_pos[last_key_id][1]
        piano_center = (y_min + y_max) / 2.0
        piano_half_length = (y_max - y_min) / 2.0

        # z location of the hands
        key_surface_z = (
            model.body_pos[first_key_id][2] + 0.01125
        )  # white keys are at z=0.01125
        forearm_z = key_surface_z + hover_offset

        # forearm ideas
        rh_forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rh_forearm")
        lh_forearm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lh_forearm")

        # set forearm positions
        model.body_pos[rh_forearm_id] = [
            forward_offset,
            piano_center + piano_half_length / 3.0,
            forearm_z,
        ]
        model.body_pos[lh_forearm_id] = [
            forward_offset,
            piano_center - piano_half_length / 3.0,
            forearm_z,
        ]

        # ids for ty
        rh_ty_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_ty")
        lh_ty_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_ty")

        # set the right hand range to the ends of the piano
        model.jnt_range[rh_ty_id] = [
            y_min - model.body_pos[rh_forearm_id][1],
            y_max - model.body_pos[rh_forearm_id][1],
        ]

        # set the left hand range to the ends of the piano
        model.jnt_range[lh_ty_id] = [
            y_min - model.body_pos[lh_forearm_id][1],
            y_max - model.body_pos[lh_forearm_id][1],
        ]

        # set control range of actuators to match joint range
        for prefix in ["rh", "lh"]:
            for suffix in ["tx", "ty", "tz"]:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_forearm_{suffix}"
                )
                act_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}_A_forearm_{suffix}"
                )
                model.actuator_ctrlrange[act_id] = model.jnt_range[joint_id]

        # floor is only visual, it shouldn't block key depression
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 0

        # initial forward pass to recompute positions from pre-updated positions
        mujoco.mj_forward(model, data)

        return model, data, y_min, y_max

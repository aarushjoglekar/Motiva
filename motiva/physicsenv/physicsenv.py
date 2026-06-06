import mujoco
import os
import numpy as np
import physicsenv.constants as constants
import helpers.helpers as helpers
import glfw

class PhysicsEnv:
    def __init__(self):
        # instantiation
        self.model, self.data = self.initialize_models()
        self.viewer = None
        
        # forearm body ids
        self.forearm_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_forearm")
            for hand in constants.HANDS
        ]

        # tz actuator and joint ids
        self.rz_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_forearm_tz")
        self.lz_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lh_A_forearm_tz")
        self.rz_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_tz")
        self.lz_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_tz")

        # ty joint ids
        ry_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_ty")
        ly_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_ty")

        # forearm body rescaling
        rh_base = self.model.body_pos[self.forearm_ids[0]]
        lh_base = self.model.body_pos[self.forearm_ids[1]]

        # x min/max is constant because it does not move
        forearm_pos_min = np.array([
            rh_base[0], rh_base[1] + self.model.jnt_range[ry_joint_id, 0], rh_base[2] + self.model.jnt_range[self.rz_joint_id, 0],
            lh_base[0], lh_base[1] + self.model.jnt_range[ly_joint_id, 0], lh_base[2] + self.model.jnt_range[self.lz_joint_id, 0],
        ])
        forearm_pos_max = np.array([
            rh_base[0], rh_base[1] + self.model.jnt_range[ry_joint_id, 1], rh_base[2] + self.model.jnt_range[self.rz_joint_id, 1],
            lh_base[0], lh_base[1] + self.model.jnt_range[ly_joint_id, 1], lh_base[2] + self.model.jnt_range[self.lz_joint_id, 1],
        ])

        self.forearm_pos_scale, self.forearm_pos_offset = helpers.make_rescaler(
            forearm_pos_min,
            forearm_pos_max,
            np.zeros(len(forearm_pos_min)) - 1,
            np.ones(len(forearm_pos_min))
        )

        # tz actuator and joint ids
        self.rz_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_forearm_tz")
        self.lz_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lh_A_forearm_tz")
        self.rz_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_tz")
        self.lz_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_tz")

        # gains for the tz joints in both arms
        self.z_kP = 3
        self.z_kD = 0.7
        self.z_kS = 0.313

        # range of actions
        self.action_lows  = self.model.actuator_ctrlrange[:, 0]
        self.action_highs = self.model.actuator_ctrlrange[:, 1]

        # piano joint/site ids
        black_keys = { 1, 4, 6, 9, 11 }

        self.piano_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, key_name)
            for key_name in [
                f"{'black' if i % 12 in black_keys else 'white'}_joint_{i}"
                for i in range(88)
            ]
        ]

        self.piano_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in [
                f"{'black' if i % 12 in black_keys else 'white'}_key_site_{i}"
                for i in range(88)
            ]
        ], dtype=int)

        # piano joint rescaling
        self.piano_scale, self.piano_offset = helpers.make_rescaler(
            self.model.jnt_range[self.piano_joint_ids, 0],
            self.model.jnt_range[self.piano_joint_ids, 1],
            np.zeros(len(self.piano_joint_ids)) - 1,
            np.ones(len(self.piano_joint_ids))
        ) 

        # hand joint ids
        self.hand_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in [
                f"{hand}_{joint}" for hand in constants.HANDS for joint in constants.JOINTS
            ]
        ]

        # hand joint rescaling
        self.hand_joint_scale, self.hand_joint_offset = helpers.make_rescaler(
            self.model.jnt_range[self.hand_joint_ids, 0],
            self.model.jnt_range[self.hand_joint_ids, 1],
            np.zeros(len(self.hand_joint_ids)) - 1,
            np.ones(len(self.hand_joint_ids))
        )

        # finger site ids
        self.finger_site_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, finger_site)
            for finger_site in constants.FINGER_SITE
        ], dtype=int)

    # action is a list indexed by actuator id of position values from -1 to 1
    # step will automatically scale based on each control range
    def step(self, action: np.ndarray):
        # scale actions
        scaled_action = self.action_lows + (action + 1) * 0.5 * (self.action_highs - self.action_lows)

        # pd position control
        self.data.ctrl[:] = scaled_action

        # position to force control for motors
        r_target_pos = scaled_action[self.rz_actuator_id]
        r_current_pos = self.data.qpos[self.rz_joint_id]
        r_current_vel = self.data.qvel[self.rz_joint_id]

        l_target_pos = scaled_action[self.lz_actuator_id]
        l_current_pos = self.data.qpos[self.lz_joint_id]
        l_current_vel = self.data.qvel[self.lz_joint_id]

        self.data.ctrl[self.rz_actuator_id] = self.z_kP * (r_target_pos - r_current_pos) - self.z_kD * r_current_vel + self.z_kS
        self.data.ctrl[self.lz_actuator_id] = self.z_kP * (l_target_pos - l_current_pos) - self.z_kD * l_current_vel + self.z_kS

        mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def get_obs(self):
        # qpos -> all joint positions (piano keys + each hand)
        # xpos -> forearm positions
        return (
            helpers.rescale(self.data.qpos[self.piano_joint_ids], self.piano_scale, self.piano_offset), 
            helpers.rescale(self.data.qpos[self.hand_joint_ids], self.hand_joint_scale, self.hand_joint_offset),
            helpers.rescale(self.data.xpos[self.forearm_ids].ravel(), self.forearm_pos_scale, self.forearm_pos_offset)
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

    def viewer_running(self):
        return self.viewer is None or self.viewer.is_running()

    def initialize_models(self, hover_offset:float=0.12, forward_offset:float=0.4):
        DIR = os.path.dirname(os.path.abspath(__file__))
        model = mujoco.MjModel.from_xml_path(os.path.join(DIR, "models/world.xml"))
        data = mujoco.MjData(model)

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
            piano_center + piano_half_length / 2.0,
            forearm_z,
        ]
        model.body_pos[lh_forearm_id] = [
            forward_offset,
            piano_center - piano_half_length / 2.0,
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

        # actuator ids
        rh_ty_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_forearm_ty"
        )
        lh_ty_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lh_A_forearm_ty"
        )

        # set control range of actuators to match joint range
        model.actuator_ctrlrange[rh_ty_act_id] = model.jnt_range[rh_ty_id]
        model.actuator_ctrlrange[lh_ty_act_id] = model.jnt_range[lh_ty_id]

        # initial forward pass to recompute positions from pre-updated positions
        mujoco.mj_forward(model, data)

        return model, data

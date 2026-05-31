import mujoco
import os
import numpy as np

class Environment:
    def __init__(self):
        self.model, self.data = self.initialize_models()
        self.viewer = None
        self.forearm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, forearm_name) for forearm_name in ["lh_forearm", "rh_forearm"]]
        
    def step(self, action=None):
        if action is not None:
            self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self.get_obs()
    
    def get_obs(self):
        # qpos -> all joint positions (each hand + piano keys)
        # xpos -> forearm positions
        return np.concatenate(
            (self.data.qpos, self.data.xpos[self.forearm_ids].ravel())
        )
                
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data,
                show_left_ui=False, show_right_ui=False
            )
        self.viewer.sync()
      
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
    def viewer_running(self):
        return self.viewer is None or self.viewer.is_running()
                
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
                
    def initialize_models(self, hover_offset=0.12, forward_offset=0.4):
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

        # ids for tx (world y axis)
        rh_tx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rh_forearm_ty")
        lh_tx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lh_forearm_ty")

        # set the right hand range to the ends of the piano
        model.jnt_range[rh_tx_id] = [
            y_min - model.body_pos[rh_forearm_id][1],
            y_max - model.body_pos[rh_forearm_id][1],
        ]

        # set the left hand range to the ends of the piano
        model.jnt_range[lh_tx_id] = [
            y_min - model.body_pos[lh_forearm_id][1],
            y_max - model.body_pos[lh_forearm_id][1],
        ]

        # actuator ids
        rh_tx_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rh_A_forearm_ty"
        )
        lh_tx_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lh_A_forearm_ty"
        )

        # set control range of actuators to match joint range
        model.actuator_ctrlrange[rh_tx_act_id] = model.jnt_range[rh_tx_id]
        model.actuator_ctrlrange[lh_tx_act_id] = model.jnt_range[lh_tx_id]

        # initial forward pass to recompute positions from pre-updated positions
        mujoco.mj_forward(model, data)
        
        return model, data
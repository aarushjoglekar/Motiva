import numpy as np
from physicsenv import constants
from physicsenv.control_type import ControlType
import mujoco


class MotorizedJointGroup:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_type: int):
        self.model = model
        self.data = data
        self.control_type = control_type

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in [
                    f"{hand}_A_{actuated_joint_name}"
                    for hand in constants.HANDS
                    for actuated_joint_name in constants.ACTUATED_JOINT_NAMES
                ]
            ]
        )
        self.length = len(self.actuator_ids)

        transmission_type = model.actuator_trntype[self.actuator_ids]
        transmission_ids = model.actuator_trnid[self.actuator_ids, 0]
        self.is_tendon = transmission_type == mujoco.mjtTrn.mjTRN_TENDON

        self.tendon_ids = transmission_ids[self.is_tendon]
        joint_ids = transmission_ids[~self.is_tendon]

        self.qpos_adr = np.zeros(self.length, dtype=int)
        self.qvel_adr = np.zeros(self.length, dtype=int)
        self.qpos_adr[~self.is_tendon] = model.jnt_qposadr[joint_ids]
        self.qvel_adr[~self.is_tendon] = model.jnt_dofadr[joint_ids]

        self.forcerange = model.actuator_forcerange[self.actuator_ids]

        self.target = np.zeros(self.length, dtype=np.float64)

    def step(self, target: np.ndarray):
        self.target[:] = target

        position = np.empty(self.length, dtype=np.float64)
        velocity = np.empty(self.length, dtype=np.float64)

        joint_mask = ~self.is_tendon
        position[joint_mask] = self.data.qpos[self.qpos_adr[joint_mask]]
        velocity[joint_mask] = self.data.qvel[self.qvel_adr[joint_mask]]
        position[self.is_tendon] = self.data.ten_length[self.tendon_ids]
        velocity[self.is_tendon] = self.data.ten_velocity[self.tendon_ids]

        if self.control_type == ControlType.POSITION_CONTROL:
            force = (
                constants.ACTUATED_JOINT_KP_POSITION * (self.target - position)
                - constants.ACTUATED_JOINT_DAMPING_POSITION * velocity
            )
        elif self.control_type == ControlType.VELOCITY_CONTROL:
            force = constants.ACTUATED_JOINT_KP_VELOCITY * (self.target - velocity)

        force = np.clip(force, self.forcerange[:, 0], self.forcerange[:, 1])
        self.data.ctrl[self.actuator_ids] = force

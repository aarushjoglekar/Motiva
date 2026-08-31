from physicsenv.velocity_kp_tuner import tune_velocity_kp
from physicsenv import constants

if __name__ == "__main__":
    joint_idx = 23
    joint_name = constants.ACTUATED_JOINT_NAMES[joint_idx]
    
    tune_velocity_kp(joint_name, False)
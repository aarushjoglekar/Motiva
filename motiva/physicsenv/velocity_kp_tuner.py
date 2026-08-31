from physicsenv.physicsenv import PhysicsEnv
from physicsenv.control_type import ControlType
from physicsenv import constants
import numpy as np
import mujoco
import multiprocessing as mp
from collections import deque

CONTINUOUS_MOVEMENT = False
DASHBOARD_HISTORY_LEN = 300

def run_dashboard(joint_name, target_value, kp_value, data_queue, stop_event):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox

    plt.ion()
    fig, ax = plt.subplots(num=f"Velocity kP Tuner - {joint_name}")
    fig.subplots_adjust(bottom=0.3)

    try:
        fig.canvas.manager.window.tk.call("tk", "scaling", 1.0)
    except Exception:
        pass
    (target_line,) = ax.plot([], [], label="target")
    (actual_line,) = ax.plot([], [], label="actual")
    ax.legend()

    target_box = TextBox(fig.add_axes([0.2, 0.15, 0.2, 0.05]), "target vel", initial=str(target_value.value))
    kp_box = TextBox(fig.add_axes([0.6, 0.15, 0.2, 0.05]), "kP", initial=str(kp_value.value))

    def on_target_submit(text):
        try:
            target_value.value = float(text)
        except ValueError:
            pass

    def on_kp_submit(text):
        try:
            kp_value.value = float(text)
        except ValueError:
            pass

    target_box.on_submit(on_target_submit)
    kp_box.on_submit(on_kp_submit)

    steps = deque(maxlen=DASHBOARD_HISTORY_LEN)
    targets = deque(maxlen=DASHBOARD_HISTORY_LEN)
    actuals = deque(maxlen=DASHBOARD_HISTORY_LEN)

    while plt.fignum_exists(fig.number):
        updated = False
        while not data_queue.empty():
            step, target, actual = data_queue.get()
            steps.append(step)
            targets.append(target)
            actuals.append(actual)
            updated = True
        if updated:
            target_line.set_data(steps, targets)
            actual_line.set_data(steps, actuals)
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.05)

    stop_event.set()


def tune_velocity_kp(joint_name: str, is_left_hand: bool):
    print("Tuning joint: ", joint_name)
    
    joint_index = None
    for index, actuated_joint in enumerate(constants.ACTUATED_JOINT_NAMES):
        if actuated_joint == joint_name:
            joint_index = index
            break
    if joint_index is None:
        raise ValueError(f"Joint {joint_name} does not exist")

    physicsenv = PhysicsEnv(
        seed=42, control_type=ControlType.VELOCITY_CONTROL, include_dynamics_data=False
    )

    if is_left_hand:
        joint_index += physicsenv.num_actions() // 2

    hand_prefix = "lh" if is_left_hand else "rh"
    joint_id = mujoco.mj_name2id(
        physicsenv.model, mujoco.mjtObj.mjOBJ_JOINT, f"{hand_prefix}_{joint_name}"
    )
    if joint_id == -1:
        raise ValueError(f"Joint {hand_prefix}_{joint_name} does not exist")

    joint_qpos_adr = physicsenv.model.jnt_qposadr[joint_id]
    joint_range_lo, joint_range_hi = physicsenv.model.jnt_range[joint_id]
    joint_center = (joint_range_lo + joint_range_hi) / 2
    joint_edge_margin = 0.05 * (joint_range_hi - joint_range_lo)

    target_value = mp.Value("d", 0.0)
    kp_value = mp.Value("d", float(constants.ACTUATED_JOINT_KP_VELOCITY[joint_index]))
    data_queue = mp.Queue()
    stop_event = mp.Event()

    dashboard_process = mp.Process(
        target=run_dashboard,
        args=(joint_name, target_value, kp_value, data_queue, stop_event),
        daemon=True,
    )
    dashboard_process.start()

    physicsenv.reset()
    physicsenv.render()

    step = 0
    while physicsenv.viewer_running() and not stop_event.is_set():
        step += 1

        target_velocity = target_value.value
        kp = kp_value.value
        constants.ACTUATED_JOINT_KP_VELOCITY[joint_index] = kp

        action = np.zeros(physicsenv.num_actions())
        action[joint_index] = target_velocity

        for _ in range(physicsenv.physics_steps_per_env_step):
            physicsenv.motorized_joints.step(action)
            mujoco.mj_step(physicsenv.model, physicsenv.data)
        physicsenv.render()

        joint_velocity = physicsenv.data.qvel[joint_id]
        data_queue.put((step, target_velocity, joint_velocity))

        if CONTINUOUS_MOVEMENT:
            joint_qpos = physicsenv.data.qpos[joint_qpos_adr]
            if (
                joint_qpos <= joint_range_lo + joint_edge_margin
                or joint_qpos >= joint_range_hi - joint_edge_margin
            ):
                physicsenv.data.qpos[joint_qpos_adr] = joint_center
                mujoco.mj_forward(physicsenv.model, physicsenv.data)

    dashboard_process.terminate()
    dashboard_process.join()

    print(f"Tuned kP for joint {joint_name}: {kp_value.value}")

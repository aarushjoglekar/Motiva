from physicsenv.physicsenv import PhysicsEnv
from physicsenv.control_type import ControlType
from physicsenv import constants
import numpy as np
import mujoco
import multiprocessing as mp
from collections import deque

def run_dashboard(joint_name, target_value, kp_value, data_queue, stop_event, dashboard_length):
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

    steps = deque(maxlen=dashboard_length)
    targets = deque(maxlen=dashboard_length)
    actuals = deque(maxlen=dashboard_length)

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


def tune_velocity_kp(joint_name: str, is_left_hand: bool, continuous_movement:bool=False, dashboard_length:int=300):
    print("Tuning joint:", joint_name)
    
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
    actuator_id = mujoco.mj_name2id(
        physicsenv.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{hand_prefix}_A_{joint_name}"
    )
    if actuator_id == -1:
        raise ValueError(f"Actuator {hand_prefix}_A_{joint_name} does not exist")

    is_tendon = physicsenv.model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON
    transmission_id = physicsenv.model.actuator_trnid[actuator_id, 0]

    if is_tendon:
        tendon_id = transmission_id
        wrap_adr = physicsenv.model.tendon_adr[tendon_id]
        wrap_num = physicsenv.model.tendon_num[tendon_id]
        coupled_joint_ids = [
            physicsenv.model.wrap_objid[i] for i in range(wrap_adr, wrap_adr + wrap_num)
        ]

        def read_joint_velocity():
            return physicsenv.data.ten_velocity[tendon_id]
    else:
        coupled_joint_ids = [transmission_id]
        joint_dof_adr = physicsenv.model.jnt_dofadr[transmission_id]

        def read_joint_velocity():
            return physicsenv.data.qvel[joint_dof_adr]

    joint_qpos_adrs = [physicsenv.model.jnt_qposadr[jid] for jid in coupled_joint_ids]
    joint_ranges = [physicsenv.model.jnt_range[jid] for jid in coupled_joint_ids]
    joint_centers = [(lo + hi) / 2 for lo, hi in joint_ranges]
    joint_edge_margins = [0.05 * (hi - lo) for lo, hi in joint_ranges]

    target_value = mp.Value("d", 0.0)
    kp_value = mp.Value("d", float(constants.ACTUATED_JOINT_KP_VELOCITY[joint_index]))
    data_queue = mp.Queue()
    stop_event = mp.Event()

    dashboard_process = mp.Process(
        target=run_dashboard,
        args=(joint_name, target_value, kp_value, data_queue, stop_event, dashboard_length),
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

        joint_velocity = read_joint_velocity()
        data_queue.put((step, target_velocity, joint_velocity))

        if continuous_movement:
            recentered = False
            for qpos_adr, (range_lo, range_hi), center, margin in zip(
                joint_qpos_adrs, joint_ranges, joint_centers, joint_edge_margins
            ):
                qpos = physicsenv.data.qpos[qpos_adr]
                if qpos <= range_lo + margin or qpos >= range_hi - margin:
                    physicsenv.data.qpos[qpos_adr] = center
                    recentered = True
            if recentered:
                mujoco.mj_forward(physicsenv.model, physicsenv.data)

    dashboard_process.terminate()
    dashboard_process.join()

    print(f"Tuned kP for joint {joint_name}: {kp_value.value}")

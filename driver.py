from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCENE_PATH = "C:/robotics/UAM/Collected_litterRoad/litterRoad.usd"
STATS_PATH = ROOT / "episode_stats.json"
EPISODE_VARIANCE_DIR = ROOT / "episode_variance_plots"

TRASH_PATH = "/World/Cylinder_01"
TRASH_CAN_NAME = "trashCan"
DRONE_BODY_PATH = "/World/iris/body"

PHYSICS_DT = 1.0 / 60.0
MAX_EPISODES = 100
SUCCESS_RADIUS_XY = 2.0
GRIPPER_TOTAL_MASS = 1.5 + 0.12 + 0.12
SETTLE_STEPS = 120
TRASH_APPROACH_Z_OFFSET = 1.0
TRASH_CAN_DROP_Z_OFFSET = 1.5
GOAL_POSITION_TOL = 0.2
GOAL_VELOCITY_TOL = 0.2
GOAL_HOVER_STEPS = int(1.0 / PHYSICS_DT)
TAKEOFF_Z = 1.5
TAKEOFF_RISE_STEPS = int(2.0 / PHYSICS_DT)
TAKEOFF_HOLD_STEPS = int(1.0 / PHYSICS_DT)
GRASP_Z_OFFSET = 0.18
BOTTLE_LOCAL_Y_OFFSET = 0.2
DESCEND_STEPS = int(5.0 / PHYSICS_DT)
CLOSE_GRIPPER_STEPS = int(1.5 / PHYSICS_DT)
REPLAN_PROGRESS_DISTANCE = 1.5
NO_REPLAN_GOAL_DISTANCE = 2.0

POSE_MEAS_HZ = 10.0
POSE_MEAS_STEP_INTERVAL = max(1, int(round(1.0 / (POSE_MEAS_HZ * PHYSICS_DT))))
POSE_MEAS_POS_STD = 0.05
POSE_MEAS_ROT_STD_DEG = 1.0

IMU_ACCEL_BIAS_STD = 0.01
IMU_GYRO_BIAS_STD = 0.01

MAP_MIN = (-10.0, -10.0, -1.0)
MAP_MAX = (10.0, 10.0, 6.0)
MAP_VOXEL_SIZE = 0.2
MAP_INFLATION_RADIUS = 0.3

PLANNER_MAX_ACC = 3.0
PLANNER_MAX_VEL = 2.0
PLANNER_TAU = 0.5
PLANNER_HORIZON = 7.0

DRONE_START_POS = np.array([[-9.0, 5.5, 0.2]], dtype=np.float32)
DRONE_START_QUAT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
ZERO_VEL = np.zeros((1, 3), dtype=np.float32)

# The hover / belief-convergence test lives in grasp_pipeline.py.


def main():
    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import open_stage
    enable_extension("isaacsim.util.debug_draw")
    enable_extension("isaacsim.sensors.physics")

    from contact_monitor import DroneContactMonitor
    from gripper_controller import GripperController
    from gripper_osprey_sandbox import attach_osprey_gripper
    from kino_astar import KinodynamicAStar
    from low_level_controller import Drone, LowLevelController, IRIS_MASS
    from mission_manager import MissionManager
    from physics_materials import apply_high_friction_to_colliders
    from trajectory_debug import TrajectoryDebugDraw
    from world_map import VoxelWorldMap
    from belief import BeliefDrone, noisy_imu, reset_noisy_imu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    open_stage(SCENE_PATH)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
    stage = world.stage
    trash_can_prim = find_prim_by_name(stage, TRASH_CAN_NAME)
    if trash_can_prim is None:
        raise RuntimeError(f"Could not find prim named {TRASH_CAN_NAME}")
    trash_can_position = prim_world_position(trash_can_prim)

    world_map = VoxelWorldMap.from_stage(
        stage,
        map_min=MAP_MIN,
        map_max=MAP_MAX,
        voxel_size=MAP_VOXEL_SIZE,
        inflation_radius=MAP_INFLATION_RADIUS,
        exclude_prefixes=("/World/iris", "/World/gripper_mount", TRASH_PATH),
    )

    trash = RigidPrim(prim_paths_expr=TRASH_PATH, name="trash")
    drone_body = RigidPrim(prim_paths_expr=DRONE_BODY_PATH, name="drone_body")
    gripper_handles = attach_osprey_gripper(
        stage,
        drone_body_path=DRONE_BODY_PATH,
        mount_world_pos=DRONE_START_POS[0],
        animate=False,
    )
    apply_high_friction_to_colliders(
        stage,
        (
            TRASH_PATH,
            "/World/gripper_mount/left_gripper_finger/osprey_collision",
            "/World/gripper_mount/right_gripper_finger/osprey_collision",
        ),
    )
    gripper = GripperController.from_attach_result(gripper_handles)
    drone_contact_monitor = DroneContactMonitor(body_path=DRONE_BODY_PATH)

    world.reset()
    trash.initialize()
    drone_body.initialize()
    drone_contact_monitor.initialize()
    drone = Drone(drone_body, None, m=IRIS_MASS + GRIPPER_TOTAL_MASS)
    belief_drone = BeliefDrone(dt=PHYSICS_DT, initial_true_state=belief_initial_state_from_drone(drone))
    # Identical twin that never receives update_pose, for the dead-reckoning comparison.
    belief_dead_reckoned = BeliefDrone(dt=PHYSICS_DT, initial_true_state=belief_initial_state_from_drone(drone))
    reset_noisy_imu(drone, IMU_ACCEL_BIAS_STD, IMU_GYRO_BIAS_STD)

    controller = LowLevelController(drone)
    planner = KinodynamicAStar(
        occupancy_map=world_map,
        max_acc=PLANNER_MAX_ACC,
        max_vel=PLANNER_MAX_VEL,
        tau=PLANNER_TAU,
        planning_horizon=PLANNER_HORIZON,
        goal_tol=GOAL_POSITION_TOL,
    )

    trajectory_debug = TrajectoryDebugDraw(enabled=True)
    mission = MissionManager(
        world=world,
        trash=trash,
        drone_body=drone_body,
        planner=planner,
        controller=controller,
        world_map=world_map,
        trajectory_debug=trajectory_debug,
        drone_start_pos=DRONE_START_POS,
        drone_start_quat=DRONE_START_QUAT,
        zero_vel=ZERO_VEL,
        dt=PHYSICS_DT,
        settle_steps=SETTLE_STEPS,
        trash_approach_z_offset=TRASH_APPROACH_Z_OFFSET,
        trash_can_position=trash_can_position,
        trash_can_drop_z_offset=TRASH_CAN_DROP_Z_OFFSET,
        takeoff_z=TAKEOFF_Z,
        takeoff_rise_steps=TAKEOFF_RISE_STEPS,
        takeoff_hold_steps=TAKEOFF_HOLD_STEPS,
        goal_position_tol=GOAL_POSITION_TOL,
        goal_velocity_tol=GOAL_VELOCITY_TOL,
        goal_hover_steps=GOAL_HOVER_STEPS,
        grasp_z_offset=GRASP_Z_OFFSET,
        bottle_local_y_offset=BOTTLE_LOCAL_Y_OFFSET,
        descend_steps=DESCEND_STEPS,
        gripper=gripper,
        close_gripper_steps=CLOSE_GRIPPER_STEPS,
        replan_progress_distance=REPLAN_PROGRESS_DISTANCE,
        no_replan_goal_distance=NO_REPLAN_GOAL_DISTANCE,
        astar_tau=PLANNER_TAU,
        max_vel=PLANNER_MAX_VEL,
        max_episodes=MAX_EPISODES,
        success_radius_xy=SUCCESS_RADIUS_XY,
        stats_path=STATS_PATH,
    )

    clear_generated_plots(EPISODE_VARIANCE_DIR)
    episode_var_t = []
    episode_var_pos = []
    episode_var_phase = []
    episode_counter = mission.completed_episodes

    step = 0
    while simulation_app.is_running():
        world.step(render=True)

        state = drone.get_state()

        belief_active = mission.phase not in (MissionManager.RESET_EPISODE, MissionManager.SETTLE_TRASH)
        if belief_active:
            obs_accel, obs_gyro = noisy_imu(drone, dt=PHYSICS_DT, accel_noise_std=0.05, gyro_noise_std=0.05)
            belief_drone.predict_imu(obs_accel, obs_gyro)
            belief_dead_reckoned.predict_imu(obs_accel, obs_gyro)

            if step % POSE_MEAS_STEP_INTERVAL == 0:
                pos_meas, R_meas = noisy_pose_measurement(
                    state, POSE_MEAS_POS_STD, POSE_MEAS_ROT_STD_DEG
                )
                belief_drone.update_pose(
                    pos_meas, R_meas,
                    pos_std=POSE_MEAS_POS_STD,
                    rot_std_deg=POSE_MEAS_ROT_STD_DEG,
                )

        est_state = belief_state_dict(belief_drone) if belief_active else state

        if belief_active:
            episode_var_t.append(len(episode_var_t) * PHYSICS_DT)
            episode_var_pos.append([
                float(belief_drone.P[0, 0]),
                float(belief_drone.P[1, 1]),
                float(belief_drone.P[2, 2]),
            ])
            episode_var_phase.append(mission.phase)

        phase_before_update = mission.phase
        sigma, sigmad, sigmadd, sigmaddd = mission.update(est_state, step)
        if phase_before_update != mission.phase and mission.phase in (MissionManager.SETTLE_TRASH, MissionManager.TAKEOFF):
            belief_drone = BeliefDrone(dt=PHYSICS_DT, initial_true_state=belief_initial_state_from_drone(drone))
            reset_noisy_imu(drone, IMU_ACCEL_BIAS_STD, IMU_GYRO_BIAS_STD)

        belief_state = belief_drone.get_state()
        trajectory_debug.add_belief_position(belief_state[0:3])
        trajectory_debug.add_actual_position(state["pos"])

        contact_info = drone_contact_monitor.update(phase=phase_before_update, step=step)
        mission.record_contact(contact_info, phase_before_update, step)

        if mission.completed_episodes > episode_counter:
            save_position_variance_plot(
                EPISODE_VARIANCE_DIR
                / f"episode_{mission.completed_episodes:03d}_position_variance.png",
                episode_var_t,
                episode_var_pos,
                episode_var_phase,
                plt,
            )
            episode_var_t = []
            episode_var_pos = []
            episode_var_phase = []
            episode_counter = mission.completed_episodes

        u, _ = controller.PDControlLaw(
            sigma,
            sigmad,
            sigmadd,
            sigmaddd,
            dt=PHYSICS_DT,
            state=est_state,
        )
        drone.apply_control(u)

        mission.draw_debug()
        if mission.complete:
            break

        step += 1

    simulation_app.close()


def find_prim_by_name(stage, name):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def prim_world_position(prim):
    from pxr import UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
    translation = matrix.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]], dtype=float)


def hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)


def noisy_pose_measurement(state, pos_std, rot_std_deg):
    pos_meas = np.asarray(state["pos"], dtype=float) + pos_std * np.random.randn(3)
    dtheta = np.radians(rot_std_deg) * np.random.randn(3)
    R_meas = np.asarray(state["R"], dtype=float) @ expm(hat(dtheta))
    return pos_meas, R_meas


def clear_generated_plots(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def save_position_variance_plot(path, t, variances, phases, plt):
    if len(t) < 2:
        return

    t = np.asarray(t, dtype=float)
    variances = np.asarray(variances, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    for index, (label, color) in enumerate(
        (("x", "tab:blue"), ("y", "tab:orange"), ("z", "tab:green"))
    ):
        ax.semilogy(t, variances[:, index], label=f"$P_{{{label}{label}}}$",
                    color=color, linewidth=1.2)

    # Mark where the mission changed phase, so the shape can be read against flight events.
    last_phase = None
    for t_phase, phase in zip(t, phases):
        if phase != last_phase:
            ax.axvline(t_phase, color="k", linestyle=":", linewidth=0.6, alpha=0.4)
            ax.text(t_phase, ax.get_ylim()[1], f" {phase}", rotation=90,
                    va="top", ha="left", fontsize=6, alpha=0.7)
            last_phase = phase

    ax.set_title(f"Belief position variance over episode  ({len(t) * PHYSICS_DT:.1f} s)")
    ax.set_xlabel("seconds since episode start")
    ax.set_ylabel("variance (m$^2$, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def belief_state_dict(belief_drone):
    R = belief_drone.R.copy()
    return {
        "pos": belief_drone.pos.copy(),
        "vel": belief_drone.vel.copy(),
        "R": R,
        "omega": belief_drone.omega.copy(),
        "yaw": float(np.arctan2(R[1, 0], R[0, 0])),
    }


def belief_initial_state_from_drone(drone):
    state = drone.get_state()
    return np.concatenate([
        state["pos"],
        state["vel"],
        state["R"].reshape(9),
        state["omega"],
    ])


if __name__ == "__main__":
    main()

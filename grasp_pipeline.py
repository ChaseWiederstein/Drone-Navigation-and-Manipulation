"""Perception-to-grasp pipeline sandbox.

Standalone driver that bypasses MissionManager. Right now it only runs the hover
leg (moved here out of driver.py); the fake-DINO detection, stabilise and grasp
phases get added on top of the same loop.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SCENE_PATH = "C:/robotics/UAM/Collected_litterRoad/litterRoad.usd"

TRASH_PATH = "/World/Cylinder_01"
DRONE_BODY_PATH = "/World/iris/body"
CAMERA_PATH = "/World/iris/body/Gemini2_ground/Orbbec_Gemini2/camera_rgb/camera_rgb/Stream_rgb"
CAMERA_RESOLUTION = (640, 480)
TRASH_HALF = np.array([0.5, 0.5, 0.5])  # m, half-extent of the fake detection box

PHYSICS_DT = 1.0 / 60.0
GRIPPER_TOTAL_MASS = 1.5 + 0.12 + 0.12

DRONE_START_POS = np.array([[-9.0, 5.5, 0.2]], dtype=np.float32)
TRRASH_DROP_POS = np.array([[-8.0, 5.5, 0.2]], dtype=np.float32)
DRONE_START_QUAT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
ZERO_VEL = np.zeros((1, 3), dtype=np.float32)

POSE_MEAS_HZ = 10.0
POSE_MEAS_STEP_INTERVAL = max(1, int(round(1.0 / (POSE_MEAS_HZ * PHYSICS_DT))))
POSE_MEAS_POS_NOISE = 0.0
POSE_MEAS_ROT_NOISE_DEG = 0

IMU_ACCEL_BIAS_NOISE = 0.00
IMU_GYRO_BIAS_NOISE = 0.00

PIXEL_NOISE = 0.0   # px, jitter on the DINO box centre

HOVER_Z = 3.0
HOVER_YAW = 0.0
HOVER_RISE_SECONDS = 4.0
HOVER_HOLD_STEPS = 500
HOVER_RUN_FOREVER = True

HOVER_PLOT_PATH = ROOT / "hover_test_error.png"
HOVER_EST_PLOT_PATH = ROOT / "hover_test_estimation_error.png"
HOVER_DR_PLOT_PATH = ROOT / "hover_test_dead_reckoning.png"
HOVER_ZOOM_PLOT_PATH = ROOT / "hover_test_zoom_z.png"
HOVER_ZOOM_START = 200
HOVER_ZOOM_STEPS = 24
APPROACH_Z_OFFSET = 0.5   # hover this far above the detected point
DETECT_INTERVAL = 5       # sim steps between DINO calls during the descent

DETECTION_IMAGE_PATH = ROOT / "dino_detection.png"
HOVER_BAYES_PLOT_PATH = ROOT / "hover_test_bayes_z.png"
HOVER_BAYES_CYCLES = 3
HOVER_BAYES_START_SECONDS = HOVER_RISE_SECONDS + 3.0


# Pipeline phases. Only HOVER is implemented; the rest are the planned sequence.
HOVER = "hover"
DETECT = "detect"
DESCEND = "descend"      # fly to the detected point, re-detecting on the way
HOLD = "hold"            # arrived, just hover
GRASP = "grasp"          # TODO: grasp pose + close gripper


def main():
    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import open_stage
    enable_extension("isaacsim.util.debug_draw")
    enable_extension("isaacsim.sensors.physics")
    enable_extension("isaacsim.sensors.camera")
    from isaacsim.sensors.camera import Camera

    from gripper_controller import GripperController
    from gripper_osprey_sandbox import attach_osprey_gripper
    from low_level_controller import Drone, LowLevelController, IRIS_MASS, quat_wxyz_to_matrix
    from physics_materials import apply_high_friction_to_colliders
    from trajectory_debug import TrajectoryDebugDraw
    from belief import BeliefDrone, noisy_imu, reset_noisy_imu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    open_stage(SCENE_PATH)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
    stage = world.stage

    camera = Camera(
        prim_path=CAMERA_PATH,
        resolution=CAMERA_RESOLUTION,
        frequency=20,
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

    world.reset()
    trash.initialize()
    drone_body.initialize()

    camera.initialize()
    for _ in range(3):
        world.step(render=True)        # render product needs a few frames before K is valid
    K = camera.get_intrinsics_matrix()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"camera K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    _, trash_quat = trash.get_world_poses()        # keep the USD rotation
    trash.set_world_poses(positions=TRRASH_DROP_POS, orientations=trash_quat)
    trash.set_linear_velocities(ZERO_VEL)
    trash.set_angular_velocities(ZERO_VEL)

    drone = Drone(drone_body, None, m=IRIS_MASS + GRIPPER_TOTAL_MASS)

    # Fixed camera mount, read off the sim once (the calibrated extrinsic in reality).
    cam_pos0, cam_quat0 = camera.get_world_pose()
    s0 = drone.get_state()
    R_bc = s0["R"].T @ quat_wxyz_to_matrix(cam_quat0)
    t_bc = s0["R"].T @ (np.asarray(cam_pos0, float) - s0["pos"])

    belief_drone = BeliefDrone(dt=PHYSICS_DT, initial_true_state=belief_initial_state_from_drone(drone))
    # Identical twin that never receives update_pose, for the dead-reckoning comparison.
    belief_dead_reckoned = BeliefDrone(dt=PHYSICS_DT, initial_true_state=belief_initial_state_from_drone(drone))
    reset_noisy_imu(drone, IMU_ACCEL_BIAS_NOISE, IMU_GYRO_BIAS_NOISE)

    controller = LowLevelController(drone)
    trajectory_debug = TrajectoryDebugDraw(enabled=True)

    phase = HOVER
    target = None
    descend_refs = []
    descend_ref_index = 0

    hover_t = []
    hover_true_err = []
    hover_belief_err = []
    hover_est_err = []
    hover_dr_err = []
    hover_z_err = []
    hover_z_sigma = []
    hover_dr_z_err = []
    hover_dr_z_sigma = []
    hover_meas_z_err = []
    hover_bayes = []
    hover_plot_saved = False
    measurement_applied = False

    step = 0
    while simulation_app.is_running():
        world.step(render=True)

        state = drone.get_state()

        obs_accel, obs_gyro = noisy_imu(drone, dt=PHYSICS_DT, accel_noise_std=0.0, gyro_noise_std=0.0)
        belief_drone.predict_imu(obs_accel, obs_gyro)
        belief_dead_reckoned.predict_imu(obs_accel, obs_gyro)

        measurement_applied = step % POSE_MEAS_STEP_INTERVAL == 0
        if measurement_applied:
            pos_meas, R_meas = noisy_pose_measurement(
                state, POSE_MEAS_POS_NOISE, POSE_MEAS_ROT_NOISE_DEG
            )
            z_prior = float(belief_drone.pos[2])
            z_prior_sigma = float(np.sqrt(belief_drone.P[2, 2]))
            belief_drone.update_pose(
                pos_meas, R_meas,
                pos_std=POSE_MEAS_POS_NOISE,
                rot_std_deg=POSE_MEAS_ROT_NOISE_DEG,
            )
            if (
                step * PHYSICS_DT >= HOVER_BAYES_START_SECONDS
                and len(hover_bayes) < HOVER_BAYES_CYCLES
            ):
                hover_bayes.append({
                    "z_true": float(state["pos"][2]),
                    "z_prior": z_prior,
                    "z_prior_sigma": z_prior_sigma,
                    "z_meas": float(pos_meas[2]),
                    "z_meas_sigma": POSE_MEAS_POS_NOISE,
                    "z_post": float(belief_drone.pos[2]),
                    "z_post_sigma": float(np.sqrt(belief_drone.P[2, 2])),
                })

        est_state = belief_state_dict(belief_drone)

        if phase == HOVER:
            sigma, sigmad, sigmadd, sigmaddd = hover_reference(
                step * PHYSICS_DT,
                DRONE_START_POS[0],
                HOVER_Z,
                HOVER_RISE_SECONDS,
                HOVER_YAW,
            )
        elif phase == DESCEND:
            sigma, sigmad, sigmadd, sigmaddd = descend_refs[
                min(descend_ref_index, len(descend_refs) - 1)
            ]
            descend_ref_index += 1
        elif phase == HOLD:
            sigma = np.array([target[0], target[1], target[2], HOVER_YAW], dtype=float)
            sigmad = np.zeros(4, dtype=float)
            sigmadd = np.zeros(4, dtype=float)
            sigmaddd = np.zeros(4, dtype=float)
        else:
            sigma, sigmad, sigmadd, sigmaddd = hold_reference(est_state)

        trash_true = trash.get_world_poses()[0][0]

        if phase == HOVER and step == 300:
            detection, point_est = detect_target(
                trash, K, camera, est_state, R_bc, t_bc, trash_true[2]
            )
            if detection is not None:
                save_detection_image(camera, detection, DETECTION_IMAGE_PATH, plt)
                target = point_est + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
                descend_refs = build_descend_refs(est_state, target, HOVER_YAW)
                descend_ref_index = 0
                print(f"[dino] first fix {np.round(point_est, 3)} "
                      f"err={np.linalg.norm(point_est - trash_true):.4f} m "
                      f"-> target {np.round(target, 3)}")
                phase = DESCEND
            else:
                phase = DETECT

        elif phase == DESCEND and step % DETECT_INTERVAL == 0:
            detection, point_est = detect_target(
                trash, K, camera, est_state, R_bc, t_bc, trash_true[2]
            )
            if detection is not None:
                new_target = point_est + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
                if np.linalg.norm(new_target - target) < 0.3:      # outlier gate
                    target = new_target
                else:
                    print(f"[dino] rejected jump of "
                          f"{np.linalg.norm(new_target - target):.3f} m")

            # Replan from where we are now; only the first slice ever gets flown.
            descend_refs = build_descend_refs(est_state, target, HOVER_YAW)
            descend_ref_index = 0

            remaining = float(np.linalg.norm(est_state["pos"] - target))
            if step % 30 == 0:
                if point_est is None:
                    target_err = "no detection"
                else:
                    target_err = f"{np.linalg.norm(point_est - trash_true):.4f} m"
                print(f"[descend] alt={est_state['pos'][2]:5.2f} "
                      f"remaining={remaining:5.3f} target_err={target_err}")

            if remaining < 0.1:
                print(f"[descend] arrived, holding at {np.round(est_state['pos'], 3)}")
                phase = HOLD

        belief_state = belief_drone.get_state()
        trajectory_debug.add_belief_position(belief_state[0:3])
        trajectory_debug.add_actual_position(state["pos"])

        u, _ = controller.PDControlLaw(
            sigma,
            sigmad,
            sigmadd,
            sigmaddd,
            dt=PHYSICS_DT,
            state=est_state,
        )
        drone.apply_control(u)

        if step == 0:
            # Red reference line: straight up from the start pose to the hover target.
            trajectory_debug.plan_points = [
                (float(DRONE_START_POS[0, 0]), float(DRONE_START_POS[0, 1]), float(DRONE_START_POS[0, 2])),
                (float(DRONE_START_POS[0, 0]), float(DRONE_START_POS[0, 1]), float(HOVER_Z)),
            ]
        trajectory_debug.draw()

        if step * PHYSICS_DT >= HOVER_RISE_SECONDS and not hover_plot_saved:
            hover_t.append(len(hover_t) * PHYSICS_DT)
            hover_true_err.append(float(np.linalg.norm(state["pos"] - sigma[0:3])))
            hover_belief_err.append(float(np.linalg.norm(est_state["pos"] - sigma[0:3])))
            hover_est_err.append(float(np.linalg.norm(state["pos"] - est_state["pos"])))
            hover_dr_err.append(float(np.linalg.norm(state["pos"] - belief_dead_reckoned.pos)))
            hover_z_err.append(float(belief_drone.pos[2] - state["pos"][2]))
            hover_z_sigma.append(float(np.sqrt(belief_drone.P[2, 2])))
            hover_dr_z_err.append(float(belief_dead_reckoned.pos[2] - state["pos"][2]))
            hover_dr_z_sigma.append(float(np.sqrt(belief_dead_reckoned.P[2, 2])))
            hover_meas_z_err.append(
                float(pos_meas[2] - state["pos"][2]) if measurement_applied else float("nan")
            )

        if step % 60 == 0:
            print(
                f"t={step * PHYSICS_DT:6.1f}s "
                f"phase={phase} "
                f"ref_z={sigma[2]:6.3f} "
                f"true_z={state['pos'][2]:6.3f} "
                f"est_z={est_state['pos'][2]:6.3f} "
                f"track_err={np.linalg.norm(est_state['pos'] - sigma[0:3]):6.3f} "
                f"est_err={np.linalg.norm(est_state['pos'] - state['pos']):6.3f} "
                f"ba={np.round(belief_drone.ba, 4)} "
                f"bg={np.round(belief_drone.bg, 4)}"
            )

        if len(hover_true_err) >= HOVER_HOLD_STEPS and not hover_plot_saved:
            save_hover_error_plot(
                HOVER_PLOT_PATH, hover_t, hover_true_err, hover_belief_err, plt
            )
            save_hover_estimation_error_plot(
                HOVER_EST_PLOT_PATH, hover_t, hover_est_err, plt
            )
            save_dead_reckoning_plot(
                HOVER_DR_PLOT_PATH, hover_t, hover_est_err, hover_dr_err, plt
            )
            save_zoom_z_plot(
                HOVER_ZOOM_PLOT_PATH,
                hover_t, hover_z_err, hover_z_sigma,
                hover_dr_z_err, hover_dr_z_sigma, hover_meas_z_err,
                HOVER_ZOOM_START, HOVER_ZOOM_STEPS, plt,
            )
            save_bayes_z_plot(HOVER_BAYES_PLOT_PATH, hover_bayes, plt)
            print(
                f"hover done, wrote {HOVER_PLOT_PATH}, "
                f"{HOVER_EST_PLOT_PATH}, {HOVER_DR_PLOT_PATH}, "
                f"{HOVER_ZOOM_PLOT_PATH}, {HOVER_BAYES_PLOT_PATH}"
            )
            hover_plot_saved = True
            if not HOVER_RUN_FOREVER:
                break


        step += 1

    simulation_app.close()

def build_descend_refs(state, target, yaw):
    """Smooth B-spline from the current state to `target`, arriving stopped.

    Rebuilt from scratch on every detection, so only the first slice ever gets flown.
    """
    from bspline import bspline_refs_from_kino_path

    pos = np.asarray(state["pos"], dtype=float)
    target = np.asarray(target, dtype=float)
    duration = float(np.clip(np.linalg.norm(target - pos) / 0.5, 1.0, 6.0))

    kino_path = [
        (pos.copy(), np.asarray(state["vel"], dtype=float).copy()),
        (target.copy(), np.zeros(3, dtype=float)),
    ]
    refs, _, _ = bspline_refs_from_kino_path(
        kino_path,
        controller_dt=PHYSICS_DT,
        astar_tau=duration,
        max_vel=2.0,
        yaw=yaw,
        requested_control_points=12,
        terminal_position=target,
        terminal_velocity=np.zeros(3, dtype=float),
        terminal_acceleration=np.zeros(3, dtype=float),
    )
    return refs


def detect_target(trash, K, camera, est_state, R_bc, t_bc, z_ground):
    """One DINO call -> world point of the trash, using the *believed* camera pose."""
    detection = DINO(trash.get_world_poses()[0][0], K, camera)
    if detection is None:
        return None, None

    R_wc_est = est_state["R"] @ R_bc
    p_wc_est = est_state["pos"] + est_state["R"] @ t_bc
    return detection, unproject(detection["box"], K, R_wc_est, p_wc_est, z_ground)


def unproject(box, K, R_wc, p_wc, z_ground):
    """Box center -> world point, by intersecting the camera ray with z = z_ground.

    Pass the believed camera pose to get the target the drone thinks it sees;
    pass the true pose to get where it actually is.
    """
    W, H = CAMERA_RESOLUTION
    u, v = box[0] * W, box[1] * H
    d_cv = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d_cam = np.array([d_cv[2], -d_cv[0], -d_cv[1]])      # OpenCV -> Isaac fwd/left/up
    ray = R_wc @ d_cam
    s = (z_ground - p_wc[2]) / ray[2]
    return p_wc + s * ray


def DINO(trash_pos, K, camera):
    from low_level_controller import quat_wxyz_to_matrix

    cam_pos, cam_quat = camera.get_world_pose()
    R = quat_wxyz_to_matrix(cam_quat)

    c = np.asarray(trash_pos, float)
    corners = np.array([c + TRASH_HALF * np.array([sx, sy, sz])
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    # Isaac's camera frame is [forward, left, up]; OpenCV wants [right, down, forward].
    q = (R.T @ (corners - np.asarray(cam_pos, float)).T).T   # (8, 3) forward/left/up
    p = np.stack([-q[:, 1], -q[:, 2], q[:, 0]], axis=1)      # -> right/down/forward

    if np.any(p[:, 2] <= 1e-3):
        print("[dino] reject: corner behind camera")
        return None

    us = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    vs = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]

    u0, v0, u1, v1 = us.min(), vs.min(), us.max(), vs.max()
    box_cx = (u0 + u1) / 2 + PIXEL_NOISE * np.random.randn()
    box_cy = (v0 + v1) / 2 + PIXEL_NOISE * np.random.randn()
    box_w, box_h = u1 - u0, v1 - v0

    W, H = CAMERA_RESOLUTION
    if not (0 <= box_cx < W and 0 <= box_cy < H):
        print("[dino] reject: center outside frame")
        return None

    return {"box": [box_cx / W, box_cy / H, box_w / W, box_h / H],
            "score": 0.9, "label": "bottle"}



def save_detection_image(camera, detection, path, plt):
    rgba = camera.get_rgba()
    if rgba is None or rgba.size == 0:
        print("[dino] no frame available")
        return

    W, H = CAMERA_RESOLUTION
    box_cx, box_cy, box_w, box_h = detection["box"]
    x0 = (box_cx - box_w / 2.0) * W
    y0 = (box_cy - box_h / 2.0) * H

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.imshow(rgba[:, :, :3])
    ax.add_patch(plt.Rectangle((x0, y0), box_w * W, box_h * H,
                               fill=False, edgecolor="lime", linewidth=2))
    ax.plot(box_cx * W, box_cy * H, marker="+", color="lime", markersize=12)
    ax.set_title(f"{detection['label']}  {detection['score']:.2f}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"[dino] wrote {path}")


def list_cameras(stage):
    """Print every Camera prim and the world direction it looks along (USD: -Z)."""
    from pxr import UsdGeom

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Camera":
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        view_dir = -np.array([m[2][0], m[2][1], m[2][2]], dtype=float)
        print(f"[cam] {prim.GetPath().pathString}  looks {np.round(view_dir, 3)}")


def find_camera_path(stage, name):
    """Path of the Camera prim whose path contains `name` (the named prim is
    usually an Xform wrapping the real camera)."""
    cameras = [p.GetPath().pathString for p in stage.Traverse()
               if p.GetTypeName() == "Camera"]
    for path in cameras:
        if name in path:
            return path
    raise RuntimeError(f"no Camera prim matching {name!r}; cameras on stage: {cameras}")


def hover_reference(t, start_pos, target_z, rise_seconds, yaw):
    """Smooth rise from start_pos to target_z over rise_seconds, then hold forever.

    Returns flat-output references (x, y, z, yaw) and their first three derivatives.
    The cosine blend keeps position, velocity and acceleration continuous so the
    controller's feedforward terms stay well behaved.
    """
    x = float(start_pos[0])
    y = float(start_pos[1])
    z0 = float(start_pos[2])
    dz = float(target_z) - z0
    T = float(rise_seconds)

    s = min(1.0, max(0.0, float(t) / T))
    if s >= 1.0:
        z, zd, zdd, zddd = z0 + dz, 0.0, 0.0, 0.0
    else:
        z = z0 + dz * (0.5 - 0.5 * np.cos(np.pi * s))
        zd = dz * 0.5 * np.pi * np.sin(np.pi * s) / T
        zdd = dz * 0.5 * np.pi ** 2 * np.cos(np.pi * s) / T ** 2
        zddd = -dz * 0.5 * np.pi ** 3 * np.sin(np.pi * s) / T ** 3

    sigma = np.array([x, y, z, float(yaw)], dtype=float)
    sigmad = np.array([0.0, 0.0, zd, 0.0], dtype=float)
    sigmadd = np.array([0.0, 0.0, zdd, 0.0], dtype=float)
    sigmaddd = np.array([0.0, 0.0, zddd, 0.0], dtype=float)
    return sigma, sigmad, sigmadd, sigmaddd


def hold_reference(state):
    sigma = np.array([state["pos"][0], state["pos"][1], state["pos"][2], state["yaw"]], dtype=float)
    zero = np.zeros(4, dtype=float)
    return sigma, zero.copy(), zero.copy(), zero.copy()


def hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)


def noisy_pose_measurement(state, pos_std, rot_std_deg):
    pos_meas = np.asarray(state["pos"], dtype=float) + pos_std * np.random.randn(3)
    dtheta = np.radians(rot_std_deg) * np.random.randn(3)
    R_meas = np.asarray(state["R"], dtype=float) @ expm(hat(dtheta))
    return pos_meas, R_meas


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


def save_hover_error_plot(path, t, true_error, belief_error, plt):
    t = np.asarray(t, dtype=float)
    true_error = np.asarray(true_error, dtype=float)
    belief_error = np.asarray(belief_error, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t, true_error, label="truth - reference", linewidth=1.2)
    ax.plot(t, belief_error, label="belief - reference", linewidth=1.0, alpha=0.7)
    ax.set_title(
        f"Hover position error  (mean {true_error.mean():.4f} m, max {true_error.max():.4f} m)"
    )
    ax.set_xlabel("seconds after reaching target height")
    ax.set_ylabel("position error norm (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_hover_estimation_error_plot(path, t, est_error, plt):
    t = np.asarray(t, dtype=float)
    est_error = np.asarray(est_error, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(t, est_error, label="truth - belief", color="tab:red", linewidth=1.2)
    ax.set_title(
        f"Hover estimation error  (mean {est_error.mean():.4f} m, max {est_error.max():.4f} m)"
    )
    ax.set_xlabel("seconds after reaching target height")
    ax.set_ylabel("position error norm (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_dead_reckoning_plot(path, t, corrected_error, dr_error, plt):
    t = np.asarray(t, dtype=float)
    corrected_error = np.asarray(corrected_error, dtype=float)
    dr_error = np.asarray(dr_error, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.semilogy(t, dr_error, label="dead reckoning (IMU only)", color="tab:red", linewidth=1.4)
    ax.semilogy(t, corrected_error, label=f"corrected ({POSE_MEAS_HZ:.0f} Hz pose fixes)",
                color="tab:blue", linewidth=1.4)
    ax.set_title(
        "Position estimation error during hover — same IMU stream, same noise\n"
        f"dead reckoning ends at {dr_error[-1]:.3f} m, corrected mean {corrected_error.mean():.4f} m"
    )
    ax.set_xlabel("seconds after reaching target height")
    ax.set_ylabel("|truth - belief|  (m, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_zoom_z_plot(path, t, z_err, z_sigma, dr_z_err, dr_z_sigma, meas_z_err, start, count, plt):
    stop = start + count
    if len(t) < stop:
        start, stop = 0, min(count, len(t))

    t = np.asarray(t[start:stop], dtype=float)
    z_err = np.asarray(z_err[start:stop], dtype=float)
    z_sigma = np.asarray(z_sigma[start:stop], dtype=float)
    dr_z_err = np.asarray(dr_z_err[start:stop], dtype=float)
    dr_z_sigma = np.asarray(dr_z_sigma[start:stop], dtype=float)
    meas_z_err = np.asarray(meas_z_err[start:stop], dtype=float)
    meas_mask = np.isfinite(meas_z_err)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)

    for ax, err, sigma, title, color in (
        (axes[0], z_err, z_sigma, f"Corrected — {POSE_MEAS_HZ:.0f} Hz pose fixes", "tab:blue"),
        (axes[1], dr_z_err, dr_z_sigma, "Dead reckoning — IMU only", "tab:red"),
    ):
        ax.fill_between(t, -sigma, sigma, color=color, alpha=0.18, label="filter $\\pm 1\\sigma$")
        ax.plot(t, err, color=color, linewidth=1.6, marker="o", markersize=3,
                label="belief z $-$ truth z")
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
        ax.set_title(title)
        ax.set_ylabel("z error (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    for t_meas in t[meas_mask]:
        axes[0].axvline(t_meas, color="k", linestyle="--", linewidth=0.7, alpha=0.5)
    axes[0].scatter(t[meas_mask], meas_z_err[meas_mask], color="k", zorder=5, s=22,
                    label="measurement")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("seconds after reaching target height")
    fig.suptitle("Covariance collapse at each measurement (z axis)", y=0.99)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _gaussian(x, mu, sigma):
    sigma = max(float(sigma), 1e-9)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def save_bayes_z_plot(path, cycles, plt):
    if not cycles:
        return

    fig, axes = plt.subplots(len(cycles), 1, figsize=(10, 3.0 * len(cycles)), dpi=150)
    if len(cycles) == 1:
        axes = [axes]

    for i, (ax, c) in enumerate(zip(axes, cycles), start=1):
        spread = max(c["z_prior_sigma"], c["z_meas_sigma"], c["z_post_sigma"])
        centre = c["z_true"]
        lo = min(centre, c["z_prior"], c["z_meas"], c["z_post"]) - 4.0 * spread
        hi = max(centre, c["z_prior"], c["z_meas"], c["z_post"]) + 4.0 * spread
        x = np.linspace(lo, hi, 800)

        ax.plot(x, _gaussian(x, c["z_prior"], c["z_prior_sigma"]),
                color="tab:blue", linewidth=1.6,
                label=f"prior  $\\sigma$={c['z_prior_sigma']:.4f}")
        ax.plot(x, _gaussian(x, c["z_meas"], c["z_meas_sigma"]),
                color="tab:orange", linewidth=1.6, linestyle="--",
                label=f"measurement  $\\sigma$={c['z_meas_sigma']:.4f}")
        ax.fill_between(x, _gaussian(x, c["z_post"], c["z_post_sigma"]),
                        color="tab:green", alpha=0.25)
        ax.plot(x, _gaussian(x, c["z_post"], c["z_post_sigma"]),
                color="tab:green", linewidth=2.0,
                label=f"posterior  $\\sigma$={c['z_post_sigma']:.4f}")
        ax.axvline(c["z_true"], color="k", linewidth=1.2, label="truth")

        ax.set_title(f"Update {i}", loc="left", fontsize=10)
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("z position (m)")
    fig.suptitle("Bayesian update on z:  prior $\\times$ measurement $\\rightarrow$ posterior", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

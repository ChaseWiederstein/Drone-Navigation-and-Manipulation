import numpy as np
import time
import json
from pathlib import Path

from bspline import bspline_refs_from_kino_path
from episode import sample_trash_position
from low_level_controller import quat_wxyz_to_matrix


ZERO4 = np.zeros(4, dtype=float)


class MissionManager:
    RESET_EPISODE = "reset_episode"
    SETTLE_TRASH = "settle_trash"
    TAKEOFF = "takeoff"
    PLAN_TO_TRASH = "plan_to_trash"
    TRACK_TO_TRASH = "track_to_trash"
    TERMINAL_HOVER = "terminal_hover"
    DESCEND_TO_TRASH = "descend_to_trash"
    CLOSE_GRIPPER = "close_gripper"
    LIFT_AFTER_GRASP = "lift_after_grasp"
    PLAN_TO_BIN = "plan_to_bin"
    TRACK_TO_BIN = "track_to_bin"
    BIN_HOVER = "bin_hover"
    RELEASE_GRIPPER = "release_gripper"

    def __init__(
        self,
        *,
        world,
        trash,
        drone_body,
        planner,
        controller,
        world_map,
        trajectory_debug,
        drone_start_pos,
        drone_start_quat,
        zero_vel,
        dt,
        settle_steps=120,
        trash_approach_z_offset=1.0,
        trash_can_position=None,
        trash_can_drop_z_offset=1.0,
        takeoff_z=1.5,
        takeoff_rise_steps=120,
        takeoff_hold_steps=60,
        goal_position_tol=0.15,
        goal_velocity_tol=0.2,
        goal_hover_steps=60,
        grasp_z_offset=0.15,
        bottle_local_y_offset=0.05,
        descend_steps=180,
        gripper=None,
        close_gripper_steps=90,
        release_gripper_steps=60,
        replan_progress_distance=1.5,
        no_replan_goal_distance=2.0,
        astar_tau=0.5,
        max_vel=2.0,
        max_episodes=100,
        success_radius_xy=5.0,
        max_consecutive_plan_failures=5,
        stats_path="episode_stats.json",
    ):
        self.world = world
        self.trash = trash
        self.drone_body = drone_body
        self.planner = planner
        self.controller = controller
        self.world_map = world_map
        self.trajectory_debug = trajectory_debug
        self.drone_start_pos = np.asarray(drone_start_pos, dtype=np.float32).reshape(1, 3)
        self.drone_start_quat = np.asarray(drone_start_quat, dtype=np.float32).reshape(1, 4)
        self.zero_vel = np.asarray(zero_vel, dtype=np.float32).reshape(1, 3)
        self.dt = float(dt)
        self.settle_steps = int(settle_steps)
        self.trash_approach_z_offset = float(trash_approach_z_offset)
        if trash_can_position is None:
            trash_can_position = np.array([0.0, 0.0, takeoff_z], dtype=float)
        self.trash_can_position = np.asarray(trash_can_position, dtype=float).reshape(3)
        self.trash_can_drop_z_offset = float(trash_can_drop_z_offset)
        self.bin_goal = self.trash_can_position.copy()
        self.bin_goal[2] += self.trash_can_drop_z_offset
        self.takeoff_z = float(takeoff_z)
        self.takeoff_rise_steps = int(takeoff_rise_steps)
        self.takeoff_hold_steps = int(takeoff_hold_steps)
        self.goal_position_tol = float(goal_position_tol)
        self.goal_velocity_tol = float(goal_velocity_tol)
        self.goal_hover_steps = int(goal_hover_steps)
        self.grasp_z_offset = float(grasp_z_offset)
        self.bottle_local_y_offset = float(bottle_local_y_offset)
        self.descend_steps = int(descend_steps)
        self.gripper = gripper
        self.close_gripper_steps = int(close_gripper_steps)
        self.release_gripper_steps = int(release_gripper_steps)
        self.replan_progress_distance = float(replan_progress_distance)
        self.no_replan_goal_distance = float(no_replan_goal_distance)
        self.astar_tau = float(astar_tau)
        self.max_vel = float(max_vel)
        self.max_episodes = int(max_episodes)
        self.success_radius_xy = float(success_radius_xy)
        self.max_consecutive_plan_failures = int(max_consecutive_plan_failures)
        self.stats_path = Path(stats_path)

        self.episode_index = 0
        self.completed_episodes = 0
        self.complete = False
        self.phase = self.RESET_EPISODE
        self.phase_step = 0
        self.goal = None
        self.trash_position = None
        self.pickup_position = None
        self.grasp_goal = None
        self.lift_start = None
        self.descend_start = None
        self.descend_refs = []
        self.descend_ref_index = 0
        self.grasp_yaw = 0.0
        self.trash_y_yaw = 0.0
        self.refs = []
        self.ref_index = 0
        self.hover_steps = 0
        self.distance_tracked_since_replan = 0.0
        self.last_ref_pos = None
        self.reported_goal_hold = False
        self.reported_descend_complete = False
        self.reported_lift_complete = False
        self.reported_bin_hold = False
        self.reported_release_complete = False
        self.episode_success = False
        self.episode_contact_happened = False
        self.episode_contacts = []
        self.episode_planning_times = []
        self.trash_bin_dist_xy = float("inf")
        self.consecutive_plan_failures = 0
        self.all_episode_results = []
        self.all_planning_times = []

    def update(self, state, step):
        if self.complete:
            return hold_reference(state)

        self._update_success_flag()
        self.trajectory_debug.add_actual_position(state["pos"])

        if self.phase == self.RESET_EPISODE:
            self._reset_episode()

        if self.phase == self.SETTLE_TRASH:
            return self._update_settle(state)

        if self.phase == self.TAKEOFF:
            return self._update_takeoff(state, step)

        if self.phase == self.PLAN_TO_TRASH:
            if self._plan(state, step):
                self._plan_succeeded()
                self._transition(self.TRACK_TO_TRASH)
            else:
                self._plan_failed("trash")
                return hold_reference(state)

        if self.phase == self.TRACK_TO_TRASH:
            return self._update_track(state, step)

        if self.phase == self.TERMINAL_HOVER:
            return self._update_terminal_hover(state, step)

        if self.phase == self.DESCEND_TO_TRASH:
            return self._update_descend_to_trash(state, step)

        if self.phase == self.CLOSE_GRIPPER:
            return self._update_close_gripper(state, step)

        if self.phase == self.LIFT_AFTER_GRASP:
            return self._update_lift_after_grasp(state, step)

        if self.phase == self.PLAN_TO_BIN:
            self.goal = self.bin_goal.copy()
            if self._plan(state, step, label="bin_planner"):
                self._plan_succeeded()
                self._transition(self.TRACK_TO_BIN)
            else:
                self._plan_failed("bin")
                return hold_reference(state)

        if self.phase == self.TRACK_TO_BIN:
            return self._update_track_to_bin(state, step)

        if self.phase == self.BIN_HOVER:
            return self._update_bin_hover(state, step)

        if self.phase == self.RELEASE_GRIPPER:
            return self._update_release_gripper(state, step)

        return hold_reference(state)

    def record_contact(self, contact_info, phase, step):
        if self.complete or not contact_info.get("new_contact", False):
            return

        contact_failure_phases = {
            self.TRACK_TO_TRASH,
            self.LIFT_AFTER_GRASP,
            self.TRACK_TO_BIN,
            self.BIN_HOVER,
            self.RELEASE_GRIPPER,
        }
        if phase not in contact_failure_phases:
            return

        contact_record = {
            "step": int(step),
            "phase": str(phase),
            "force": float(contact_info.get("force", 0.0)),
            "number_of_contacts": int(contact_info.get("number_of_contacts", 0)),
            "contacts": list(contact_info.get("contacts", [])),
        }
        self.episode_contact_happened = True
        self.episode_contacts.append(contact_record)
        self._finish_episode(reason="contact")

    def draw_debug(self):
        self.trajectory_debug.draw()

    def _reset_episode(self):
        if self.completed_episodes >= self.max_episodes:
            self.complete = True
            return

        self.episode_index += 1
        trash_position = sample_trash_position()
        _, current_trash_orientation = self.trash.get_world_poses()
        self.trash.set_world_poses(
            positions=trash_position.reshape(1, 3).astype(np.float32),
            orientations=current_trash_orientation,
        )
        self.trash.set_linear_velocities(self.zero_vel)
        self.trash.set_angular_velocities(self.zero_vel)
        self._reset_drone_pose()
        self.controller.integral_pos_error[:] = 0.0
        self.trajectory_debug.clear_all()
        self.refs = []
        self.ref_index = 0
        self.hover_steps = 0
        self.distance_tracked_since_replan = 0.0
        self.last_ref_pos = None
        self.reported_goal_hold = False
        self.reported_descend_complete = False
        self.reported_lift_complete = False
        self.reported_bin_hold = False
        self.reported_release_complete = False
        self.episode_success = False
        self.episode_contact_happened = False
        self.episode_contacts = []
        self.episode_planning_times = []
        self.trash_bin_dist_xy = float("inf")
        self.consecutive_plan_failures = 0
        if self.gripper is not None:
            self.gripper.open()
        self._transition(self.SETTLE_TRASH)

    def _update_settle(self, state):
        if self.phase_step < self.settle_steps:
            self.phase_step += 1
            return hold_reference(state)

        trash_positions, trash_orientations = self.trash.get_world_poses()
        settled = np.asarray(trash_positions[0], dtype=float)
        trash_quat = np.asarray(trash_orientations[0], dtype=float)
        trash_y_world = _normalized_xy(quat_wxyz_to_matrix(trash_quat) @ np.array([0.0, 1.0, 0.0]))
        self.trash_position = settled.copy()
        self.pickup_position = settled + self.bottle_local_y_offset * trash_y_world
        self.goal = self.pickup_position.copy()
        self.goal[2] += self.trash_approach_z_offset
        self.grasp_goal = self.pickup_position.copy()
        self.grasp_goal[2] += self.grasp_z_offset
        self.trash_y_yaw = _yaw_of_world_axis(trash_y_world)
        self.grasp_yaw = _wrap_angle(_yaw_for_body_y_axis(trash_y_world) + np.pi / 2.0)
        self._reset_drone_pose()
        self.controller.integral_pos_error[:] = 0.0
        self._transition(self.TAKEOFF)
        return hold_reference(state)

    def _update_takeoff(self, state, step):
        alpha = min(1.0, self.phase_step / max(1, self.takeoff_rise_steps))
        takeoff_z = float(self.drone_start_pos[0, 2]) + alpha * (self.takeoff_z - float(self.drone_start_pos[0, 2]))
        sigma = np.array([self.drone_start_pos[0, 0], self.drone_start_pos[0, 1], takeoff_z, state["yaw"]], dtype=float)
        self.phase_step += 1

        if self.phase_step >= self.takeoff_rise_steps + self.takeoff_hold_steps:
            self.controller.integral_pos_error[:] = 0.0
            self._transition(self.PLAN_TO_TRASH)

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _update_track(self, state, step):
        goal_distance = float(np.linalg.norm(state["pos"] - self.goal))
        if goal_distance <= self.no_replan_goal_distance:
            self._transition(self.TERMINAL_HOVER)
            return self._update_terminal_hover(state, step)

        if self.distance_tracked_since_replan >= self.replan_progress_distance:
            self._transition(self.PLAN_TO_TRASH)
            if self._plan(state, step):
                self._plan_succeeded()
                self._transition(self.TRACK_TO_TRASH)
            else:
                self._plan_failed("trash")
                return hold_reference(state)

        if self.refs:
            ref = self.refs[min(self.ref_index, len(self.refs) - 1)]
            self.ref_index += 1
        else:
            self._transition(self.PLAN_TO_TRASH)
            return hold_reference(state)

        self._accumulate_reference_distance(ref[0][0:3])
        return ref

    def _update_terminal_hover(self, state, step):
        sigma = np.array([self.goal[0], self.goal[1], self.goal[2], state["yaw"]], dtype=float)
        goal_error = float(np.linalg.norm(state["pos"] - self.goal))
        speed = float(np.linalg.norm(state["vel"]))
        if goal_error <= self.goal_position_tol and speed <= self.goal_velocity_tol:
            self.hover_steps += 1
        else:
            self.hover_steps = 0

        if self.hover_steps >= self.goal_hover_steps and not self.reported_goal_hold:
            self.reported_goal_hold = True
            self.descend_start = np.asarray(state["pos"], dtype=float).copy()
            self._build_descend_refs(state)
            self.controller.integral_pos_error[:] = 0.0
            self._transition(self.DESCEND_TO_TRASH)

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _update_descend_to_trash(self, state, step):
        if self.descend_refs:
            sigma, sigmad, sigmadd, sigmaddd = self.descend_refs[min(self.descend_ref_index, len(self.descend_refs) - 1)]
            self.descend_ref_index += 1
        else:
            sigma = np.array([self.grasp_goal[0], self.grasp_goal[1], self.grasp_goal[2], self.grasp_yaw], dtype=float)
            sigmad = ZERO4.copy()
            sigmadd = ZERO4.copy()
            sigmaddd = ZERO4.copy()

        self.phase_step += 1

        descend_complete = self.descend_refs and self.descend_ref_index >= len(self.descend_refs)
        if descend_complete:
            if not self.reported_descend_complete:
                self.reported_descend_complete = True
            speed = float(np.linalg.norm(state["vel"]))
            if speed <= self.goal_velocity_tol:
                self.controller.integral_pos_error[:] = 0.0
                self._transition(self.CLOSE_GRIPPER)

        return sigma, sigmad, sigmadd, sigmaddd

    def _update_close_gripper(self, state, step):
        if self.phase_step == 0:
            pass

        if self.gripper is not None:
            close_fraction = min(1.0, (self.phase_step + 1) / max(1, self.close_gripper_steps))
            self.gripper.close_fraction(close_fraction)

        sigma = np.array([self.grasp_goal[0], self.grasp_goal[1], self.grasp_goal[2], self.grasp_yaw], dtype=float)
        self.phase_step += 1

        if self.phase_step == self.close_gripper_steps:
            self.lift_start = np.asarray(state["pos"], dtype=float).copy()
            self.controller.integral_pos_error[:] = 0.0
            self._transition(self.LIFT_AFTER_GRASP)

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _update_lift_after_grasp(self, state, step):
        if self.gripper is not None:
            self.gripper.hold()

        if self.lift_start is None:
            self.lift_start = np.asarray(state["pos"], dtype=float).copy()

        alpha = min(1.0, self.phase_step / max(1, self.takeoff_rise_steps))
        z = float(self.lift_start[2]) + alpha * (self.takeoff_z - float(self.lift_start[2]))
        sigma = np.array([self.lift_start[0], self.lift_start[1], z, self.grasp_yaw], dtype=float)
        self.phase_step += 1

        if self.phase_step >= self.takeoff_rise_steps + self.takeoff_hold_steps:
            if not self.reported_lift_complete:
                self.reported_lift_complete = True
                self.goal = self.bin_goal.copy()
                self.refs = []
                self.ref_index = 0
                self.hover_steps = 0
                self.distance_tracked_since_replan = 0.0
                self.last_ref_pos = None
                self.controller.integral_pos_error[:] = 0.0
                self._transition(self.PLAN_TO_BIN)
            sigma[2] = self.takeoff_z

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _update_track_to_bin(self, state, step):
        if self.gripper is not None:
            self.gripper.hold()

        goal_distance = float(np.linalg.norm(state["pos"] - self.goal))
        if goal_distance <= self.no_replan_goal_distance:
            self.hover_steps = 0
            self._transition(self.BIN_HOVER)
            return self._update_bin_hover(state, step)

        if self.distance_tracked_since_replan >= self.replan_progress_distance:
            self._transition(self.PLAN_TO_BIN)
            if self._plan(state, step, label="bin_planner"):
                self._plan_succeeded()
                self._transition(self.TRACK_TO_BIN)
            else:
                self._plan_failed("bin")
                return hold_reference(state)

        if self.refs:
            ref = self.refs[min(self.ref_index, len(self.refs) - 1)]
            self.ref_index += 1
        else:
            self._transition(self.PLAN_TO_BIN)
            return hold_reference(state)

        self._accumulate_reference_distance(ref[0][0:3])
        return ref

    def _update_bin_hover(self, state, step):
        if self.gripper is not None:
            self.gripper.hold()

        sigma = np.array([self.bin_goal[0], self.bin_goal[1], self.bin_goal[2], self.grasp_yaw], dtype=float)
        goal_error = float(np.linalg.norm(state["pos"] - self.bin_goal))
        speed = float(np.linalg.norm(state["vel"]))
        if goal_error <= self.goal_position_tol and speed <= self.goal_velocity_tol:
            self.hover_steps += 1
        else:
            self.hover_steps = 0

        if self.hover_steps >= self.goal_hover_steps and not self.reported_bin_hold:
            self.reported_bin_hold = True
            self.controller.integral_pos_error[:] = 0.0
            self._transition(self.RELEASE_GRIPPER)

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _update_release_gripper(self, state, step):
        if self.phase_step == 0:
            if self.gripper is not None:
                self.gripper.open()

        if self.gripper is not None:
            self.gripper.hold()

        sigma = np.array([self.bin_goal[0], self.bin_goal[1], self.bin_goal[2], self.grasp_yaw], dtype=float)
        self.phase_step += 1

        if self.phase_step >= self.release_gripper_steps and not self.reported_release_complete:
            self.reported_release_complete = True
            self._finish_episode(reason="release")

        return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()

    def _build_descend_refs(self, state):
        duration = max(self.descend_steps * self.dt, self.dt)
        kino_path = [
            (np.asarray(state["pos"], dtype=float).copy(), np.zeros(3, dtype=float)),
            (self.grasp_goal.copy(), np.zeros(3, dtype=float)),
        ]
        self.descend_refs, control_points, debug = bspline_refs_from_kino_path(
            kino_path,
            controller_dt=self.dt,
            astar_tau=duration,
            max_vel=self.max_vel,
            yaw=self.grasp_yaw,
            requested_control_points=12,
            terminal_position=self.grasp_goal,
            terminal_velocity=np.zeros(3, dtype=float),
            terminal_acceleration=np.zeros(3, dtype=float),
        )
        self.descend_ref_index = 0
        self.trajectory_debug.set_plan_from_refs(self.descend_refs)

    def _plan(self, state, step, label="planner"):
        start = np.asarray(state["pos"], dtype=float)
        goal = np.asarray(self.goal, dtype=float)

        astar_t0 = time.perf_counter()
        kino_path = self.planner.search(
            start_pos=start,
            start_vel=state["vel"],
            goal_pos=goal,
            goal_vel=np.zeros(3, dtype=float),
        )
        astar_time = time.perf_counter() - astar_t0
        profile = self.planner.profile or {}
        status = profile.get("status", "unknown")
        iters = int(profile.get("iterations", 0))
        children = int(profile.get("children_attempted", 0))
        collision_pruned = int(profile.get("collision_pruned", 0))

        if kino_path is None:
            self.episode_planning_times.append(astar_time)
            self.all_planning_times.append(astar_time)
            self.refs = []
            self.ref_index = 0
            self.trajectory_debug.clear_plan()
            return False

        self.trajectory_debug.set_kino_path(kino_path)
        kino_terminal_pos = np.asarray(kino_path[-1][0], dtype=float)
        if status == "success":
            terminal_position = self.goal
        else:
            terminal_position = kino_terminal_pos
        bspline_t0 = time.perf_counter()
        self.refs, control_points, debug = bspline_refs_from_kino_path(
            kino_path,
            controller_dt=self.dt,
            astar_tau=self.astar_tau,
            max_vel=self.max_vel,
            yaw=state["yaw"],
            yaw_mode="velocity_body_y",
            terminal_position=terminal_position,
            terminal_velocity=np.zeros(3, dtype=float),
            terminal_acceleration=np.zeros(3, dtype=float),
        )
        bspline_time = time.perf_counter() - bspline_t0
        plan_time = astar_time + bspline_time
        self.episode_planning_times.append(plan_time)
        self.all_planning_times.append(plan_time)

        self.ref_index = 0
        self.distance_tracked_since_replan = 0.0
        self.last_ref_pos = None
        self.trajectory_debug.set_plan_from_refs(self.refs)
        return True

    def _plan_succeeded(self):
        self.consecutive_plan_failures = 0

    def _plan_failed(self, target_name):
        self.consecutive_plan_failures += 1
        if self.consecutive_plan_failures >= self.max_consecutive_plan_failures:
            self.episode_success = False
            self._finish_episode(reason=f"planning_failed_{target_name}")

    def _update_success_flag(self):
        trash_positions, _ = self.trash.get_world_poses()
        trash_pos = np.asarray(trash_positions[0], dtype=float)
        dist_xy = float(np.linalg.norm(trash_pos[:2] - self.trash_can_position[:2]))
        self.trash_bin_dist_xy = dist_xy
        if dist_xy <= self.success_radius_xy:
            self.episode_success = True

    def _finish_episode(self, reason):
        if self.phase == self.RESET_EPISODE:
            return

        self.completed_episodes += 1
        episode_avg_plan = float(np.mean(self.episode_planning_times)) if self.episode_planning_times else float("nan")
        self.all_episode_results.append(
            {
                "episode": self.episode_index,
                "success": bool(self.episode_success),
                "contact": bool(self.episode_contact_happened),
                "planning_times": list(self.episode_planning_times),
                "contacts": list(self.episode_contacts),
                "reason": reason,
                "trash_bin_dist_xy": self.trash_bin_dist_xy,
            }
        )

        successes = sum(1 for result in self.all_episode_results if result["success"])
        success_pct = 100.0 * successes / max(1, len(self.all_episode_results))
        if self.all_planning_times:
            overall_avg = float(np.mean(self.all_planning_times))
            overall_std = float(np.std(self.all_planning_times))
        else:
            overall_avg = float("nan")
            overall_std = float("nan")

        print(
            f"episode={self.episode_index} success={self.episode_success} "
            f"trash_made_it={self.episode_success} trash_bin_dist_xy={self.trash_bin_dist_xy:.3f} "
            f"contacts={self.episode_contact_happened} avg_planning_time={episode_avg_plan:.4f}s"
        )
        self._write_stats(success_pct, overall_avg, overall_std)

        if self.completed_episodes >= self.max_episodes:
            print(
                f"final episodes={self.completed_episodes} success_pct={success_pct:.1f} "
                f"avg_planning_time={overall_avg:.4f}s std_planning_time={overall_std:.4f}s"
            )
            self.complete = True
            return

        self._transition(self.RESET_EPISODE)

    def _write_stats(self, success_pct, overall_avg, overall_std):
        data = {
            "completed_episodes": self.completed_episodes,
            "max_episodes": self.max_episodes,
            "success_pct": success_pct,
            "avg_planning_time": overall_avg,
            "std_planning_time": overall_std,
            "episodes": self.all_episode_results,
        }
        self.stats_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _accumulate_reference_distance(self, ref_pos):
        ref_pos = np.asarray(ref_pos, dtype=float)
        if self.last_ref_pos is not None:
            self.distance_tracked_since_replan += float(np.linalg.norm(ref_pos - self.last_ref_pos))
        self.last_ref_pos = ref_pos.copy()

    def _reset_drone_pose(self):
        self.drone_body.set_world_poses(positions=self.drone_start_pos, orientations=self.drone_start_quat)
        self.drone_body.set_linear_velocities(self.zero_vel)
        self.drone_body.set_angular_velocities(self.zero_vel)

    def _transition(self, phase):
        self.phase = phase
        self.phase_step = 0


def hold_reference(state):
    sigma = np.array([state["pos"][0], state["pos"][1], state["pos"][2], state["yaw"]], dtype=float)
    return sigma, ZERO4.copy(), ZERO4.copy(), ZERO4.copy()


def _fmt_vec(v):
    arr = np.asarray(v, dtype=float).reshape(-1)
    return tuple(round(float(x), 3) for x in arr[:3])


def _yaw_for_body_y_axis(axis_world):
    axis = _normalized_xy(axis_world)
    return float(np.arctan2(axis[1], axis[0]) - np.pi / 2.0)


def _yaw_of_world_axis(axis_world):
    axis = _normalized_xy(axis_world)
    return float(np.arctan2(axis[1], axis[0]))


def _normalized_xy(axis_world):
    axis = np.asarray(axis_world, dtype=float).reshape(3)
    axis[2] = 0.0
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return axis / norm


def _body_y_axis_from_yaw(yaw):
    yaw = float(yaw)
    return np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)


def _wrap_angle(angle):
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)

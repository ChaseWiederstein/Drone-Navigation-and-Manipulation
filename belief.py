import numpy as np
from scipy.linalg import expm


def closest_rotation(R):
    R = np.asarray(R, dtype=float)
    U, _, Vt = np.linalg.svd(R)
    R_so3 = U @ Vt
    if np.linalg.det(R_so3) < 0.0:
        U[:, -1] *= -1.0
        R_so3 = U @ Vt
    return R_so3


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def log_so3(R):
    R = closest_rotation(R)
    cos_angle = 0.5 * (np.trace(R) - 1.0)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-10:
        return vee(0.5 * (R - R.T))

    return vee((angle / (2.0 * np.sin(angle))) * (R - R.T))


def bounded_vector(v, max_norm):
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        return np.zeros(3, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > float(max_norm):
        return v * (float(max_norm) / norm)
    return v


class BeliefDrone:
    def __init__(self, dt, initial_true_state=None):
        self.dt = float(dt)
        self.g = np.array([0.0, 0.0, 9.81])

        # Will be initialized later
        self.pos = initial_true_state[0:3].copy()
        self.vel = initial_true_state[3:6].copy()
        self.R = initial_true_state[6:15].reshape(3, 3).copy()
        self.omega = initial_true_state[15:18].copy()

        # --- Estimated Biases ---
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        # --- Initial uncertainty, as standard deviations ---
        self.pos_std_init = 1e-2    # m
        self.vel_std_init = 1e-3    # m/s
        self.rot_std_init = 1e-3    # rad
        self.ba_std_init = 0.01     # m/s^2   how wrong the accel bias guess may be
        self.bg_std_init = 0.01     # rad/s   how wrong the gyro bias guess may be

        # Our initial estimate of the error between nominal and truth cov
        self.P = np.diag([
            self.pos_std_init ** 2, self.pos_std_init ** 2, self.pos_std_init ** 2,
            self.vel_std_init ** 2, self.vel_std_init ** 2, self.vel_std_init ** 2,
            self.rot_std_init ** 2, self.rot_std_init ** 2, self.rot_std_init ** 2,
            self.ba_std_init ** 2, self.ba_std_init ** 2, self.ba_std_init ** 2,
            self.bg_std_init ** 2, self.bg_std_init ** 2, self.bg_std_init ** 2,
        ])

        # --- Process noise, as standard deviations ---
        self.a_obs_noise_guess = .05    # m/s^2
        self.w_obs_noise_guess = .05    # rad/s
        self.ba_drift_guess = 1e-3      # m/s^2 / sqrt(s)
        self.bg_drift_guess = 1e-4      # rad/s / sqrt(s)

        self.max_accel_norm = 50.0
        self.max_gyro_norm = 20.0

        # Guessed sensor covariance matrix
        self.Q = np.diag([
            self.a_obs_noise_guess ** 2, self.a_obs_noise_guess ** 2, self.a_obs_noise_guess ** 2,
            # Accel (x,y,z) covariance
            self.w_obs_noise_guess ** 2, self.w_obs_noise_guess ** 2, self.w_obs_noise_guess ** 2,
            # Gyro (x,y,z) covariance
            self.ba_drift_guess ** 2, self.ba_drift_guess ** 2, self.ba_drift_guess ** 2,
            # Accel bias variance
            self.bg_drift_guess ** 2, self.bg_drift_guess ** 2, self.bg_drift_guess ** 2,
            # Gyro bias variance
        ])


    def hat(self, w):
        wx, wy, wz = w
        return np.array([
            [0, -wz, wy],
            [wz, 0, -wx],
            [-wy, wx, 0]
        ], dtype=float)

    def predict_imu(self, a_obs, w_obs):
        dt = self.dt

        # 1. Correct measurements using estimated biases
        w_hat = bounded_vector(w_obs - self.bg, self.max_gyro_norm)
        a_hat = bounded_vector(a_obs - self.ba, self.max_accel_norm)

        # Store omega directly in class state
        self.omega = w_hat.copy()

        # 2. Update Nominal State (Position, Velocity, Orientation)
        a_world = (self.R @ a_hat) - self.g

        self.pos += self.vel * dt + 0.5 * a_world * (dt ** 2)
        self.vel += a_world * dt
        self.R = closest_rotation(self.R @ expm(dt * self.hat(w_hat)))

        F_k = np.eye(15)
        F_k[0:3, 3:6] = np.eye(3) * dt # 1. Position error grows from Velocity error (pos = pos + vel * dt)
        F_k[3:6, 6:9] = -self.R @ self.hat(a_hat) * dt # 2. Velocity error grows from Rotation error (accel vector points in wrong direction)
        F_k[3:6, 9:12] = -self.R * dt # 3. Velocity error grows from Accel Bias error (wrong bias shift in IMU reading)
        F_k[6:9, 12:15] = -np.eye(3) * dt # 4. Rotation error grows from Gyro Bias error (wrong gyro bias corrupts integrated angle)

        # 4. Build Noise Mapping Matrix G (15x12) & Discrete Noise Qd
        G = np.zeros((15, 12))
        G[3:6, 0:3] = -self.R  # Accel noise affects velocity
        G[6:9, 3:6] = -np.eye(3)  # Gyro noise affects rotation
        G[9:12, 6:9] = np.eye(3)  # Accel bias drift
        G[12:15, 9:12] = np.eye(3)  # Gyro bias drift

        Qd = G @ self.Q @ G.T * dt

        self.P = F_k @ self.P @ F_k.T + Qd
        self.P = 0.5 * (self.P + self.P.T)

        return self.get_state()

    def update_pose(self, pos_meas, R_meas, pos_std=2.0, rot_std_deg=10.0, pose_only=False):
        pos_meas = np.asarray(pos_meas, dtype=float)
        R_meas = closest_rotation(R_meas)

        residual = np.zeros(6)
        residual[0:3] = pos_meas - self.pos
        residual[3:6] = log_so3(self.R.T @ R_meas)

        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 6:9] = np.eye(3)

        rot_std = np.radians(float(rot_std_deg))
        R_cov = np.diag(
            [
                pos_std**2,
                pos_std**2,
                pos_std**2,
                rot_std**2,
                rot_std**2,
                rot_std**2,
            ]
        )

        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)
        if pose_only:
            K[3:6, :] = 0.0
            K[9:15, :] = 0.0

        dx = K @ residual
        self._inject_error_state(dx)

        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_cov @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        post_residual = np.zeros(6)
        post_residual[0:3] = pos_meas - self.pos
        post_residual[3:6] = log_so3(self.R.T @ R_meas)
        return residual, post_residual

    def _inject_error_state(self, dx):
        self.pos += dx[0:3]
        self.vel += dx[3:6]
        self.R = closest_rotation(self.R @ expm(self.hat(dx[6:9])))
        self.ba += dx[9:12]
        self.bg += dx[12:15]

    def get_state(self):
        return np.concatenate([
            self.pos,
            self.vel,
            self.R.reshape(9),
            self.omega,
        ])


def reset_noisy_imu(drone, accel_bias_std=0.01, gyro_bias_std=0.01):
    state = drone.get_state()
    noisy_imu._prev_vel = np.asarray(state["vel"], dtype=float).copy()
    noisy_imu._ba = accel_bias_std * np.random.randn(3)
    noisy_imu._bg = gyro_bias_std * np.random.randn(3)


def noisy_imu(
    drone,
    dt=1.0 / 60.0,
    accel_noise_std=0.00,
    gyro_noise_std=0.00,
    max_accel_norm=50.0,
    max_gyro_norm=20.0,
):
    state = drone.get_state()
    R = state["R"]
    vel = np.asarray(state["vel"], dtype=float)
    omega = state["omega"]

    prev_vel = getattr(noisy_imu, "_prev_vel", None)
    if prev_vel is None:
        prev_vel = vel.copy()
    acc_world = (vel - prev_vel) / float(dt)
    noisy_imu._prev_vel = vel.copy()

    e3 = np.array([0.0, 0.0, 1.0])
    g = float(getattr(drone, "g", 9.81))

    ba = getattr(noisy_imu, "_ba", np.zeros(3))
    bg = getattr(noisy_imu, "_bg", np.zeros(3))

    true_accel_body = bounded_vector(R.T @ (acc_world + g * e3), max_accel_norm)
    obs_accel = true_accel_body + ba + accel_noise_std * np.random.randn(3)
    obs_gyro = bounded_vector(omega, max_gyro_norm) + bg + gyro_noise_std * np.random.randn(3)

    return obs_accel, obs_gyro

import numpy as np


IRIS_MASS = 1.6186146
# IRIS_INERTIA = np.diag([0.0291249994, 0.0291249994, 0.0552249998])
IRIS_INERTIA = np.diag([3.45374723e-02, 3.14913822e-02 ,6.29682146e-02])
IRIS_KF = 8.54858e-6
IRIS_KM = 1e-6
IRIS_MIN_ROTOR_SPEED = 0.0
IRIS_MAX_ROTOR_SPEED = 1100.0
IRIS_ROTOR_DIRECTIONS = np.array([-1.0, -1.0, 1.0, 1.0], dtype=float)
IRIS_ROTOR_POSITIONS = np.array(
    [
        [0.1379854530, -0.2067163885, 0.0230000000],
        [-0.1251116842, 0.2187541425, 0.0230000000],
        [0.1379999965, 0.2025757730, 0.0230000000],
        [-0.1241535991, -0.2223457992, 0.0230000000],
    ],
    dtype=float,
)

from scipy.linalg import expm

def normalize(v, eps=1e-8):
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def quat_wxyz_to_matrix(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def yaw_from_R(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def matrix_to_quat_wxyz(R):
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def _iris_wrench_limits_from_rotors(
    rotor_positions=IRIS_ROTOR_POSITIONS,
    kf=IRIS_KF,
    km=IRIS_KM,
    rotor_directions=IRIS_ROTOR_DIRECTIONS,
    omega_max=IRIS_MAX_ROTOR_SPEED,
):
    rotor_positions = np.asarray(rotor_positions, dtype=float)
    x = rotor_positions[:, 0]
    y = rotor_positions[:, 1]
    rotor_directions = np.asarray(rotor_directions, dtype=float)

    omega2 = float(omega_max) ** 2
    f_max = 4.0 * float(kf) * omega2

    tx_pos = float(np.sum(y[y > 0.0])) * float(kf) * omega2
    tx_neg = float(np.sum(-y[y < 0.0])) * float(kf) * omega2
    tx_max = min(tx_pos, tx_neg)

    ty_pos = float(np.sum(-x[x < 0.0])) * float(kf) * omega2
    ty_neg = float(np.sum(x[x > 0.0])) * float(kf) * omega2
    ty_max = min(ty_pos, ty_neg)

    tz_pos = float(np.sum(rotor_directions[rotor_directions > 0.0])) * float(km) * omega2
    tz_neg = float(np.sum(-rotor_directions[rotor_directions < 0.0])) * float(km) * omega2
    tz_max = min(tz_pos, tz_neg)

    return float(f_max), np.array([tx_max, ty_max, tz_max], dtype=float)


IRIS_MAX_THRUST, IRIS_MAX_TORQUE = _iris_wrench_limits_from_rotors()

class TrueDrone():
    def __init__(self,body, dt, initial_state=None, mass=None, inertia=None):
        self.body = body
        self.m = float(IRIS_MASS if mass is None else mass)
        self.J = np.asarray(IRIS_INERTIA if inertia is None else inertia, dtype=float).copy()
        self.J_inv = np.linalg.inv(self.J)
        self.g = 9.81
        self.dt = float(dt)
        self.t = 0

        self.umin = np.hstack([0.0, -IRIS_MAX_TORQUE])
        self.umax = np.hstack([IRIS_MAX_THRUST, IRIS_MAX_TORQUE])

        #
        self.pos = np.array([0.0, 0.0, 3.0])
        self.vel = np.zeros(3)
        self.R = np.eye(3).reshape(9)
        self.omega = np.zeros(3)
        self.last_u = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)


        if initial_state is not None:
            self.set_from_true_state(initial_state)

    def get_state(self):
        return np.concatenate([self.pos, self.vel, self.R, self.omega])

    def vee(self, S):
        return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)

    def RotationError(self, R, Rr):
        return self.vee(0.5 * (Rr.T @ R - R.T @ Rr))

    def unpack(self, x):
        p = x[0:3]
        v = x[3:6]
        R = x[6:15].reshape(3, 3)
        w = x[15:18]
        return p, v, R, w


    def step(self, u):
        self.t += self.dt
        u = np.clip(u, self.umin, self.umax)
        self.last_u = u.copy()
        u1, tau_x, tau_y, tau_z = u

        u = np.array([u1, tau_x, tau_y, tau_z])
        self.last_u = u.copy()

        u1 = float(u[0])
        tau = np.asarray(u[1:4], dtype=float)
        dt = self.dt

        def hat(w):
            wx, wy, wz = w
            return np.array([[0, -wz, wy],
                             [wz, 0, -wx],
                             [-wy, wx, 0]], dtype=float)

        e3 = np.array([0.0, 0.0, 1.0])
        Rm = self.R.reshape(3, 3)

        zB_world = Rm @ e3
        acc = -self.g * e3 + (u1 / self.m) * zB_world

        Iw = self.J @ self.omega
        omega_dot = self.J_inv @ (-np.cross(self.omega, Iw) + tau -.05*self.omega )

        self.vel += dt * acc
        self.pos += dt * self.vel
        self.omega += dt * omega_dot

        Rm = Rm @ expm(dt * hat(self.omega))

        # U, _, Vt = np.linalg.svd(Rm)
        # Rm = U @ Vt
        # if np.linalg.det(Rm) < 0:
        #     U[:, -1] *= -1
        #     Rm = U @ Vt
        self.R = Rm.reshape(9)

        x_next = self.get_state()
        self.pushto_isaac()

        return x_next

    def set_from_true_state(self, state):
        self.set_state(np.asarray(state, dtype=float))

    def set_state(self, x):
        self.pos = x[0:3].copy()
        self.vel = x[3:6].copy()
        self.R = x[6:15].copy()
        self.omega = x[15:18].copy()

    def set_initial_state(self):
        positions, orientations = self.body.get_world_poses()
        linear_velocities = self.body.get_linear_velocities()
        angular_velocities = self.body.get_angular_velocities()

        self.pos = np.asarray(positions[0], dtype=float).copy()
        self.vel = np.asarray(linear_velocities[0], dtype=float).copy()

        quat = np.asarray(orientations[0], dtype=float)
        self.R = quat_wxyz_to_matrix(quat).reshape(9)

        omega_world = np.asarray(angular_velocities[0], dtype=float)
        self.omega = self.R.reshape(3, 3).T @ omega_world

    def pushto_isaac(self):
        Rm = self.R.reshape(3, 3)
        quat = matrix_to_quat_wxyz(Rm)

        self.body.set_world_poses(
            positions=self.pos.reshape(1, 3).astype(np.float32),
            orientations=quat.reshape(1, 4).astype(np.float32),
        )

        omega_world = Rm @ self.omega

        self.body.set_linear_velocities(self.vel.reshape(1, 3).astype(np.float32))
        self.body.set_angular_velocities(omega_world.reshape(1, 3).astype(np.float32))

    def imu_measurement(
            self,
            accel_noise_std=0.05,
            gyro_noise_std=0.01,
    ):
        Rm = self.R.reshape(3, 3)
        e3 = np.array([0.0, 0.0, 1.0])

        zB_world = Rm @ e3
        acc_world = -self.g * e3 + (self.last_u[0] / self.m) * zB_world

        true_accel = Rm.T @ (acc_world + self.g * e3)
        true_gyro= self.omega.copy()

        obs_accel = true_accel + accel_noise_std * np.random.randn(3)
        obs_gyro = true_gyro + gyro_noise_std * np.random.randn(3)

        return true_accel, obs_accel, true_gyro, obs_gyro


class TrueController:
    def __init__(
        self,
        drone,
        omegap=2.0,
        omegaR=8.0,
        max_thrust=None,
        max_torque=None,
    ):
        self.drone = drone

        self.Kp = drone.m * omegap**2 * np.eye(3)
        self.Kv = 2.0 * drone.m * omegap * np.eye(3)
        self.Kr = drone.J * omegaR**2
        self.Kw = 2.0 * drone.J * omegaR

        self.max_thrust = float(max_thrust) if max_thrust is not None else float(IRIS_MAX_THRUST)
        if max_torque is None:
            self.max_torque = np.asarray(IRIS_MAX_TORQUE, dtype=float)
        else:
            self.max_torque = np.asarray(max_torque, dtype=float)

    def PDControlLaw(self, sigma, sigmad, sigmadd, sigmaddd):
        pos = self.drone.pos
        vel = self.drone.vel
        R = self.drone.R.reshape(3, 3)
        omega = self.drone.omega

        ep = pos - sigma[0:3]
        ev = vel - sigmad[0:3]

        e3 = np.array([0.0, 0.0, 1.0])
        Fdes = -self.Kp @ ep - self.Kv @ ev + self.drone.m * self.drone.g * e3 + self.drone.m * sigmadd[0:3]

        zB = R[:, 2]
        u1 = np.dot(Fdes, zB)
        u1 = np.clip(u1, 0.0, self.max_thrust)

        zBdes = normalize(Fdes)
        if np.linalg.norm(zBdes) < 1e-8:
            zBdes = e3

        xCdes = np.array([np.cos(sigma[3]), np.sin(sigma[3]), 0.0])
        yBdes = normalize(np.cross(zBdes, xCdes))
        if np.linalg.norm(yBdes) < 1e-8:
            yBdes = np.array([0.0, 1.0, 0.0])
        xBdes = np.cross(yBdes, zBdes)

        Rdes = np.column_stack((xBdes, yBdes, zBdes))
        E = 0.5 * (Rdes.T @ R - R.T @ Rdes)
        eR = vee(E)

        a_dot = sigmaddd[0:3]
        thrust_for_reference = max(abs(u1), 1e-6)
        h = (self.drone.m / thrust_for_reference) * (a_dot - (zBdes @ a_dot) * zBdes)

        p_des = -h @ yBdes
        q_des = h @ xBdes
        r_des = sigmad[3] * (e3 @ zBdes)
        omega_des = np.array([p_des, q_des, r_des])

        ew = omega - omega_des
        torque = -self.Kr @ eR - self.Kw @ ew + np.cross(omega, self.drone.J @ omega)
        torque = np.clip(torque, -self.max_torque, self.max_torque)

        u = np.hstack([u1, torque])
        debug = {
            "ep": ep,
            "ev": ev,
            "eR": eR,
            "ew": ew,
            "Fdes": Fdes,
            "Rdes": Rdes,
        }

        return u, debug

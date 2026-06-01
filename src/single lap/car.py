"""
Car dynamics module — dynamic bicycle model for an EV race car.

Exposes step(state, delta, F_drive, dt) → CarState.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CarParams:
    """Physical parameters for the EV race car."""
    # Mass and geometry
    mass: float = 1600.0         # kg (typical Formula E + battery)
    L: float = 2.9               # wheelbase (m)
    l_f: float = 1.4             # CG to front axle (m)
    l_r: float = 1.5             # CG to rear axle (m)
    width: float = 1.8           # car width (m)
    length: float = 4.8          # car length for rendering (m)
    I_z: float = 2500.0          # yaw moment of inertia (kg*m^2)

    # Tire
    mu: float = 1.0              # tire-road friction coefficient
    C_f: float = 80000.0         # front cornering stiffness (N/rad)
    C_r: float = 90000.0         # rear cornering stiffness (N/rad)

    # Aero and rolling resistance
    C_d: float = 0.30            # drag coefficient
    A_front: float = 1.5         # frontal area (m^2)
    rho: float = 1.225           # air density (kg/m^3)
    C_roll: float = 0.012        # rolling resistance coefficient

    # Motor and battery
    P_max: float = 250e3         # max motor power (W) ~250kW
    F_drive_max: float = 8000.0  # max traction force (N)
    F_brake_max: float = 12000.0 # max braking force (N)
    eta_motor: float = 0.92      # motor efficiency
    eta_regen: float = 0.65      # regen braking efficiency
    Q_batt: float = 5.0          # battery capacity (kWh) - small pack for visible drain
    V_nom: float = 800.0         # nominal voltage (V)

    # Limits
    v_max: float = 70.0          # max speed (m/s) ~252 km/h
    delta_max: float = 0.5       # max steering angle (rad) ~28 deg
    SOC_min: float = 0.05        # min SOC


@dataclass
class CarState:
    """State of the car at a given instant."""
    x: float = 0.0       # position x (m)
    y: float = 0.0       # position y (m)
    psi: float = 0.0     # heading angle (rad)
    vx: float = 10.0     # longitudinal velocity (m/s)
    vy: float = 0.0      # lateral velocity (m/s)
    omega: float = 0.0   # yaw rate (rad/s)
    SOC: float = 1.0     # battery state of charge [0,1]

    def to_array(self):
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.omega, self.SOC])

    @staticmethod
    def from_array(arr):
        return CarState(x=arr[0], y=arr[1], psi=arr[2], vx=arr[3],
                        vy=arr[4], omega=arr[5], SOC=arr[6])

    def position(self):
        return np.array([self.x, self.y])


class BicycleModel:
    """Planar bicycle model for an EV race car."""

    def __init__(self, params=None):
        self.p = params or CarParams()

    def derivatives(self, state: CarState, delta: float, F_drive: float):
        p = self.p
        vx = max(state.vx, 0.5)
        vy = state.vy
        omega = state.omega

        # Tire slip angles
        alpha_f = delta - np.arctan2(vy + p.l_f * omega, vx)
        alpha_r = -np.arctan2(vy - p.l_r * omega, vx)

        # Lateral tire forces (linear model, capped by friction circle)
        F_yf = p.C_f * alpha_f
        F_yr = p.C_r * alpha_r
        F_yf = np.clip(F_yf, -p.mu * p.mass * 9.81 * 0.5, p.mu * p.mass * 9.81 * 0.5)
        F_yr = np.clip(F_yr, -p.mu * p.mass * 9.81 * 0.5, p.mu * p.mass * 9.81 * 0.5)

        # Clamp drive force
        F_drive = np.clip(F_drive, -p.F_brake_max, p.F_drive_max)
        if F_drive > 0 and F_drive * vx > p.P_max:
            F_drive = p.P_max / vx

        # Longitudinal forces
        F_drag = 0.5 * p.rho * p.C_d * p.A_front * vx**2
        F_roll = p.C_roll * p.mass * 9.81

        # Equations of motion
        dx = vx * np.cos(state.psi) - vy * np.sin(state.psi)
        dy = vx * np.sin(state.psi) + vy * np.cos(state.psi)
        dpsi = omega
        dvx = (F_drive - F_drag - F_roll + p.mass * vy * omega
               - F_yf * np.sin(delta)) / p.mass
        dvy = (F_yf * np.cos(delta) + F_yr - p.mass * vx * omega) / p.mass
        domega = (p.l_f * F_yf * np.cos(delta) - p.l_r * F_yr) / p.I_z

        # Battery SOC dynamics
        P_mech = F_drive * vx
        if P_mech >= 0:
            P_elec = P_mech / p.eta_motor
        else:
            P_elec = P_mech * p.eta_regen
        dSOC = -P_elec / (p.Q_batt * 3600 * 1000)

        return np.array([dx, dy, dpsi, dvx, dvy, domega, dSOC])

    def step(self, state: CarState, delta: float, F_drive: float, dt: float = 0.01):
        """Advance state by dt using RK4 integration."""
        z = state.to_array()

        def f(z_arr, d, F):
            s = CarState.from_array(z_arr)
            return self.derivatives(s, d, F)

        k1 = f(z, delta, F_drive)
        k2 = f(z + 0.5*dt*k1, delta, F_drive)
        k3 = f(z + 0.5*dt*k2, delta, F_drive)
        k4 = f(z + dt*k3, delta, F_drive)

        z_new = z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Enforce constraints
        z_new[3] = max(z_new[3], 0.1)   # min speed
        z_new[6] = np.clip(z_new[6], self.p.SOC_min, 1.0)  # SOC bounds
        z_new[2] = z_new[2] % (2 * np.pi)  # wrap heading

        return CarState.from_array(z_new)

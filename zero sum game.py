import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================================
# 1. AGENT (Simplified to Standard P-Controller)
# ======================================================================
class Agent:
    def __init__(self):
        # Constraints corresponding to h_i and b_i in the paper
        self.h_i = np.array([25.0, 35.0, 69.0])
        self.b_i = np.array([0.0, -26.0, -26.0])

        # Input mapping matrix (rho_v=1, rho_psi=10, rho_theta=10)
        self.G = np.zeros((6, 3))
        self.G[3, 0] = 1.0
        self.G[4, 1] = 10.0
        self.G[5, 2] = 10.0

    def compute_control(self, X_i, leader_state):
        """
        A basic Proportional controller to test pure kinematics.
        It drives the position error to zero without considering safety/CBF.
        """
        e_pos = X_i[0:3]
        v_r, psi_r, theta_r = leader_state

        # 1. Calculate Leader's Velocity Vector
        psi_r_rad = np.deg2rad(psi_r)
        theta_r_rad = np.deg2rad(theta_r)

        V_rx = v_r * np.cos(theta_r_rad) * np.cos(psi_r_rad)
        V_ry = v_r * np.cos(theta_r_rad) * np.sin(psi_r_rad)
        V_rz = v_r * np.sin(theta_r_rad)
        V_r_vec = np.array([V_rx, V_ry, V_rz])

        # 2. Outer Loop: P-Controller to determine desired velocity vector
        K_p = 1.0  # Proportional gain for closing the position gap
        V_des_vec = V_r_vec - K_p * e_pos

        # 3. Inner Loop: Convert desired vector to commanded states
        v_des = np.linalg.norm(V_des_vec)

        if v_des > 0.1:  # Prevent singularity when stationary
            psi_des = np.rad2deg(np.arctan2(V_des_vec[1], V_des_vec[0]))
            theta_des = np.rad2deg(np.arcsin(V_des_vec[2] / v_des))
        else:
            psi_des = psi_r
            theta_des = theta_r

        u_c = np.array([v_des, psi_des, theta_des])

        # Apply strict actuator constraints
        u_c = np.clip(u_c, self.b_i, self.h_i)

        return u_c


# ======================================================================
# 2. ENVIRONMENT (ODE Manager)
# ======================================================================
class Environment:
    def __init__(self):
        self.num_agents = 1
        self.rho_v, self.rho_psi, self.rho_theta = 1.0, 10.0, 10.0

        self.eta_ir = {
            "UAV_1": np.array([0., 0., 0.]),
        }

        self.initial_positions = {
            "UAV_1": np.array([40.0, 0.0, 600.0]),
        }

        self.agents = {}
        for i in range(1, self.num_agents + 1):
            self.agents[f"UAV_{i}"] = Agent()

    def leader_derivatives(self, t):
        if t <= 30.0:
            vr_dot = 0.0
        else:
            vr_dot = 10.0 * 0.0693 * np.exp(-0.0693 * (t - 30.0))

        psi_r_dot = 0.0

        if t < 30.0:
            theta_r_dot = 0.0
        elif 30.0 <= t <= 50.0:
            theta_r_dot = 30.0 * 0.2 * np.exp(-0.2 * (t - 30.0))
        else:
            theta_r_dot = 6.0 * (np.exp(-0.2 * (t - 30.0)) - np.exp(-0.2 * (t - 50.0)))

        return vr_dot, psi_r_dot, theta_r_dot

    def leader_state(self, t):
        if t <= 30.0:
            v_r = 10.0
        else:
            v_r = 20.0 - 10.0 * np.exp(-0.0693 * (t - 30.0))

        psi_r = 0.0

        if t < 30.0:
            theta_r = 0.0
        elif 30.0 <= t <= 50.0:
            theta_r = 30.0 * (1.0 - np.exp(-0.2 * (t - 30.0)))
        else:
            theta_r = 30.0 * (np.exp(-0.2 * (t - 50.0)) - np.exp(-0.2 * (t - 30.0)))

        return v_r, psi_r, theta_r

    def compute_kinematics(self, X_i, t):
        v_r, psi_r, theta_r = self.leader_state(t)
        vr_dot, psi_r_dot, theta_r_dot = self.leader_derivatives(t)

        v_i = X_i[3] + v_r
        psi_i = X_i[4] + psi_r
        theta_i = X_i[5] + theta_r

        psi_i_rad = np.deg2rad(psi_i)
        theta_i_rad = np.deg2rad(theta_i)
        psi_r_rad = np.deg2rad(psi_r)
        theta_r_rad = np.deg2rad(theta_r)

        f1 = v_i * np.cos(theta_i_rad) * np.cos(psi_i_rad) - v_r * np.cos(theta_r_rad) * np.cos(psi_r_rad)
        f2 = v_i * np.cos(theta_i_rad) * np.sin(psi_i_rad) - v_r * np.cos(theta_r_rad) * np.sin(psi_r_rad)
        f3 = v_i * np.sin(theta_i_rad) - v_r * np.sin(theta_r_rad)

        f4 = -self.rho_v * v_i - vr_dot
        f5 = -self.rho_psi * psi_i - psi_r_dot
        f6 = -self.rho_theta * theta_i - theta_r_dot

        F_nom = np.array([f1, f2, f3, f4, f5, f6])

        # TURNED OFF DISTURBANCE FOR KINEMATICS TEST
        d_env = np.zeros(6)

        return F_nom, d_env

    def ode_dynamics(self, t, Z):
        dot_Z = np.zeros_like(Z)

        v_r, psi_r, theta_r = self.leader_state(t)
        psi_r_rad = np.deg2rad(psi_r)
        theta_r_rad = np.deg2rad(theta_r)

        dot_Z[0] = v_r * np.cos(theta_r_rad) * np.cos(psi_r_rad)
        dot_Z[1] = v_r * np.cos(theta_r_rad) * np.sin(psi_r_rad)
        dot_Z[2] = v_r * np.sin(theta_r_rad)

        idx = 3
        for i, (name, agent) in enumerate(self.agents.items()):
            # State vector is now just X (size 6), no weights
            X = Z[idx: idx + 6]

            # Compute nominal tracking commands
            u_c = agent.compute_control(X, (v_r, psi_r, theta_r))

            # Evaluate physics
            F_nom, d_env = self.compute_kinematics(X, t)
            X_dot_physical = F_nom + agent.G @ u_c + d_env

            dot_Z[idx: idx + 6] = X_dot_physical
            idx += 6

        return dot_Z


# ======================================================================
# 3. RUNNER AND DATA EXTRACTION
# ======================================================================
def run_and_visualize():
    env = Environment()

    Z0_list = [np.array([40.0, 0.0, 600.0])]  # Leader Pos
    for name, agent in env.agents.items():
        leader_init_state = np.array([10.0, 0.0, 0.0])
        agent_init_state = np.array([10.0, 0.0, 0.0])

        e_pos = env.initial_positions[name] - env.eta_ir[name] - Z0_list[0]
        e_state = agent_init_state - leader_init_state

        X0 = np.concatenate([e_pos, e_state])
        Z0_list.append(X0)  # Only append state vector X0

    Z0 = np.concatenate(Z0_list)

    t_span = (0, 150.0)
    t_eval = np.linspace(0, 150.0, 1500)

    print("Running raw kinematics test integration...")
    sol = solve_ivp(
        fun=env.ode_dynamics,
        t_span=t_span,
        y0=Z0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-3,
        atol=1e-3
    )
    print("Integration complete. Formatting data...")

    times = sol.t
    history_pos = {a: [] for a in env.agents}
    history_ei = {a: [] for a in env.agents}
    history_v, history_psi, history_theta = {a: [] for a in env.agents}, {a: [] for a in env.agents}, {a: [] for a in
                                                                                                       env.agents}
    history_vc, history_psic, history_thetac = {a: [] for a in env.agents}, {a: [] for a in env.agents}, {a: [] for a in
                                                                                                          env.agents}

    history_leader_pos = sol.y[0:3, :].T
    history_vr, history_psir, history_thetar = [], [], []

    for t_idx, t in enumerate(times):
        v_r, psi_r, theta_r = env.leader_state(t)
        history_vr.append(v_r)
        history_psir.append(psi_r)
        history_thetar.append(theta_r)

        idx = 3
        for a_name, agent in env.agents.items():
            X_t = sol.y[idx: idx + 6, t_idx]

            u_t = agent.compute_control(X_t, (v_r, psi_r, theta_r))

            history_ei[a_name].append(X_t[:3])
            history_v[a_name].append(X_t[3] + v_r)
            history_psi[a_name].append(X_t[4] + psi_r)
            history_theta[a_name].append(X_t[5] + theta_r)

            history_vc[a_name].append(u_t[0])
            history_psic[a_name].append(u_t[1])
            history_thetac[a_name].append(u_t[2])

            pos = history_leader_pos[t_idx] + env.eta_ir[a_name] + X_t[:3]
            history_pos[a_name].append(pos)
            idx += 6

    for a in env.agents:
        history_pos[a] = np.array(history_pos[a])
        history_ei[a] = np.array(history_ei[a])
        history_v[a] = np.array(history_v[a])
        history_psi[a] = np.array(history_psi[a])
        history_theta[a] = np.array(history_theta[a])
        history_vc[a] = np.array(history_vc[a])
        history_psic[a] = np.array(history_psic[a])
        history_thetac[a] = np.array(history_thetac[a])

    plot_results(
        list(env.agents.keys()), times,
        history_pos, history_leader_pos,
        history_ei, history_v, history_psi, history_theta,
        history_vc, history_psic, history_thetac,
        np.array(history_vr), np.array(history_psir), np.array(history_thetar)
    )


# ======================================================================
# 4. PLOTTING
# ======================================================================
def plot_results(agents, times, history_pos, history_leader_pos, history_ei,
                 history_v, history_psi, history_theta, history_vc, history_psic, history_thetac,
                 history_vr, history_psir, history_thetar):
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['tab:blue']
    linestyles = ['-']

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents): ax1.plot(times, history_v[agent], color=colors[i], label=f'$v_{i + 1}$')
    ax1.plot(times, history_vr, 'k--', label='$v_r$')
    ax1.set(xlabel='Time (s)', ylabel='Velocity (m/s)')
    ax1.legend()

    fig7 = plt.figure(figsize=(12, 8))
    ax_3d = fig7.add_subplot(111, projection='3d')
    for i, agent in enumerate(agents):
        pos = history_pos[agent]
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], label=f'UAV$_{i + 1}$')
    ax_3d.plot(history_leader_pos[:, 0], history_leader_pos[:, 1], history_leader_pos[:, 2], 'k--', label='Leader')
    ax_3d.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    ax_3d.legend()

    fig8, ax8 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ei_norm = np.linalg.norm(history_ei[agent], axis=1)
        ax8.plot(times, ei_norm, color=colors[i], label=f'$\\|e_{i + 1}\\|$ (Position Error)')
    ax8.set(xlabel='Time (s)', ylabel='Tracking Error (m)')
    ax8.legend()

    os.makedirs("training_plots", exist_ok=True)
    fig1.savefig("training_plots/Fig1_Kinematics_Velocity.png")
    fig7.savefig("training_plots/Fig7_Kinematics_3D.png")
    fig8.savefig("training_plots/Fig8_Kinematics_Error.png")
    print("Kinematics plots saved!")
    plt.show()


if __name__ == "__main__":
    run_and_visualize()
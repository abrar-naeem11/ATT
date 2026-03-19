import time
import numpy as np
import pybullet as p
import pybullet_data
from Critic import CriticNetwork
import torch


class UAVSafeFormationEnv:
    def __init__(self, gui=True):

        # Environment parameters
        self.GUI = gui

        # Connect to PyBullet
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=1500,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0., 0., 600.],
                physicsClientId=self.CLIENT
            )
        else:
            self.CLIENT = p.connect(p.DIRECT)

        # Agent properties
        self.agent_mass = 0.027
        self.agent_radius = 5.0  # larger for visibility at UAV scale

        # Action Interval
        self.controller_dt = 0.01  # 100Hz Controller

        # UAV Autopilot Time Constants (Section VI)
        self.rho_v     = 1.0
        self.rho_psi   = 10.0
        self.rho_theta = 10.0

        # Desired formation offsets η_ir from Section VI
        self.eta_ir = {
            "UAV_1": np.array([0.,    0.,   0.]),
            "UAV_2": np.array([-125., -25., 0.]),
            "UAV_3": np.array([-125.,  25., 0.]),
            "UAV_4": np.array([-250., -25., 0.]),
            "UAV_5": np.array([-250.,  25., 0.]),
        }

        # Control Constraints (Section VI)
        self.b_v     = 0.0
        self.h_v     = 25.0
        self.b_psi   = -26.0
        self.h_psi   = 35.0
        self.b_theta = -26.0
        self.h_theta = 69.0

        # α_i = (h - b)/2,  β̄_i = (h + b)/2  — from Eq. 16
        self.alpha_i = np.diag([
            (self.h_v     - self.b_v)     / 2.0,
            (self.h_psi   - self.b_psi)   / 2.0,
            (self.h_theta - self.b_theta) / 2.0
        ])
        self.beta_bar_i = np.array([
            (self.h_v     + self.b_v)     / 2.0,
            (self.h_psi   + self.b_psi)   / 2.0,
            (self.h_theta + self.b_theta) / 2.0
        ])

        # RL / critic parameters (Table I)
        self.Ri                = np.eye(6)     # weight matrix R_i
        self.delta             = 2.0           # L2 gain δ
        self.kappa             = 0.1           # damping coefficient κ_i
        self.learning_rate     = 0.5           # σ_iw
        self.mini_batch_size   = 64
        self.replay_buffer_size = int(1e5)
        self.k_e               = np.array([30., 30., 20.])  # error bounds k^o_ie (Table I)

        # Agent list needed before critic init — define here so critics can be built
        self.num_agents = 5
        self.agents = [f"UAV_{i + 1}" for i in range(self.num_agents)]

        # One critic network per UAV agent
        self.critics = {
            agent: CriticNetwork(
                m=19,
                lr=self.learning_rate,
                delta=self.delta,
                kappa=self.kappa,
                replay_buffer_size=self.replay_buffer_size
            )
            for agent in self.agents
        }

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment to initial conditions from Section VI."""
        if seed is not None:
            np.random.seed(seed)

        # Reset PyBullet simulation
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems(physicsClientId=self.CLIENT)

        # Time tracking
        self.t            = 0.0
        self.step_counter = 0

        # Initial UAV positions from Section VI
        self.initial_positions = {
            "UAV_1": np.array([5.,    5.,   606.]),
            "UAV_2": np.array([-136., -37., 603.]),
            "UAV_3": np.array([-125.,  37., 596.]),
            "UAV_4": np.array([-266., -32., 595.]),
            "UAV_5": np.array([-264.,  34., 607.]),
        }

        # Virtual leader initial state from Section VI: η_r(0) = [40, 0, 600]
        self.leader_initial_pos   = np.array([40.0, 0.0, 600.0])
        self.leader_initial_state = np.array([10.0, 0.0, 0.0])   # [v_r, ψ_r, θ_r]

        # UAV internal flight states [v_i, ψ_i, θ_i] — all start at (10, 0, 0)
        self.agent_flight_states = {
            agent: np.array([10.0, 0.0, 0.0]) for agent in self.agents
        }

        self.agent_positions  = {}
        self.agent_velocities = {}

        # Reset critic weights to ones (Section VI initial condition)
        for agent in self.agents:
            with torch.no_grad():
                self.critics[agent].net.fc2.weight.data.fill_(1.0)
            self.critics[agent]._sync_w()
            self.critics[agent].replay.buf.clear()

        self._housekeeping()
        self._update_kinematic_information()

        self.leader_pos = self.leader_initial_pos.copy()
        self.leader_state = self.leader_initial_state.copy()

        return self._get_obs()

    # ------------------------------------------------------------------
    # HOUSEKEEPING
    # ------------------------------------------------------------------

    def _housekeeping(self):
        """Initialise PyBullet world and spawn UAV spheres."""
        p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.controller_dt, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        # Visual ground plane at z = 0
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2000, 2000, 0.1],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        self.PLANE_ID = p.createMultiBody(
            baseMass=0,
            basePosition=[0, 0, 0],
            baseVisualShapeIndex=plane_vis,
            physicsClientId=self.CLIENT
        )

        self.agent_ids = []

        # UAV colours for distinction
        colours = [
            [1, 0, 0, 1],   # UAV_1 red
            [0, 1, 0, 1],   # UAV_2 green
            [0, 0, 1, 1],   # UAV_3 blue
            [1, 1, 0, 1],   # UAV_4 yellow
            [1, 0, 1, 1],   # UAV_5 magenta
        ]

        for idx, agent in enumerate(self.agents):
            pos = self.initial_positions[agent].tolist()
            col_shape = p.createCollisionShape(
                p.GEOM_SPHERE, radius=self.agent_radius,
                physicsClientId=self.CLIENT
            )
            vis_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=self.agent_radius,
                rgbaColor=colours[idx],
                physicsClientId=self.CLIENT
            )
            agent_id = p.createMultiBody(
                baseMass=self.agent_mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos,
                physicsClientId=self.CLIENT
            )
            p.changeDynamics(
                agent_id, -1,
                contactProcessingThreshold=0.0,
                physicsClientId=self.CLIENT
            )
            self.agent_ids.append(agent_id)

    # ------------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------------

    def step(self):
        """
        Execute one simulation step using the critic-derived Nash policies.
        Returns: obs, reward_dict, done, truncated, info
        """
        self.t += self.controller_dt
        self.step_counter += 1

        # 1. Update virtual leader kinematics (Eq. 3)
        self._update_leader()

        # 2. Get current observations (augmented states X_i)
        obs = self._get_obs()

        actions     = {}
        reward_dict = {}

        for agent in self.agents:
            Xi = obs[agent]          # shape (6,)
            ei = Xi[:3]             # position error e_i

            # Dynamic safety bound k_ie from closest neighbour (Remark 1 / Eq. 10)
            k_ie = self._compute_k_ie(agent)

            # System matrices F_i, G_i for current state (Eq. 11)
            Fi, Gi = self._compute_FiGi(agent)

            # Critic derives approximate Nash policies û_i, d̂_i (Eq. 36, 37)
            u_hat, d_hat = self.critics[agent].compute_policy(
                Xi, ei, k_ie, Gi, self.alpha_i, self.beta_bar_i
            )

            # Clip control to constraint bounds (Section VI)
            u_hat[0] = np.clip(u_hat[0], self.b_v,     self.h_v)
            u_hat[1] = np.clip(u_hat[1], self.b_psi,   self.h_psi)
            u_hat[2] = np.clip(u_hat[2], self.b_theta, self.h_theta)

            actions[agent] = u_hat

            # Compute ẋ_i = F_i + G_i û_i + d̂_i  (used for Hamiltonian)
            # d_hat from critic is full 6D; spatial part is first 3
            Xi_dot = Fi + Gi @ u_hat + d_hat

            # Hamiltonian error and ϕ_i for weight update (Eq. 39)
            err, phi_dot = self.critics[agent].compute_hamiltonian_error(
                Xi, ei, k_ie, u_hat, d_hat, Fi, Gi,
                self.alpha_i, self.beta_bar_i, self.Ri, Xi_dot
            )

            # Augmented cost L^c_i and B̄_c_i derivative for replay storage
            B_bar, _ = self.critics[agent].compute_B_bar(ei, k_ie)
            Lc = self.critics[agent]._compute_Lc(
                Xi, u_hat, d_hat, ei, k_ie,
                self.Ri, self.alpha_i, self.beta_bar_i, B_bar
            )
            B_bar_dot = self.critics[agent].compute_dB_bar_dt(ei, k_ie, Xi_dot)

            # Update critic weights via experience replay (Eq. 42)
            self.critics[agent].update_weights(phi_dot, Lc, B_bar_dot)

            reward_dict[agent] = -float(Xi @ self.Ri @ Xi)   # negative state cost

        # 3. Apply kinematics and step PyBullet physics
        for i, agent in enumerate(self.agents):
            self._apply_uav_kinematics(agent, actions[agent], self.agent_ids[i])

        p.stepSimulation(physicsClientId=self.CLIENT)

        # 4. Update stored positions / velocities
        self._update_kinematic_information()

        # 5. Check termination: all errors within k_e bounds
        obs_new = self._get_obs()

        max_time = 150.0

        # Done ONLY if we reached the simulation time limit
        done = bool(self.t >= max_time)

        # (Optional) Gym API standards
        truncated = False

        return obs_new, reward_dict, done, False, {}

    # ------------------------------------------------------------------
    # VIRTUAL LEADER  (Eq. 3)
    # ------------------------------------------------------------------

    def _update_leader(self):
        """Integrate virtual leader position using Eq. 3."""
        vr, psi_r, theta_r = self.get_leader_state(self.t)
        self.leader_state = np.array([vr, psi_r, theta_r])

        psi_r_rad = np.deg2rad(psi_r)
        theta_r_rad = np.deg2rad(theta_r)

        self.leader_pos[0] += vr * np.cos(theta_r_rad) * np.cos(psi_r_rad) * self.controller_dt
        self.leader_pos[1] += vr * np.cos(theta_r_rad) * np.sin(psi_r_rad) * self.controller_dt
        self.leader_pos[2] += vr * np.sin(theta_r_rad) * self.controller_dt

    def get_leader_state(self, t):
        """
        Calculates the reference leader state [v_r, psi_r, theta_r]
        as a function of time (t).

        Returns:
            np.array: [velocity (m/s), heading (deg), pitch (deg)]
        """
        # 1. Reference Velocity (v_r)
        if t <= 30.0:
            v_r = 10.0
        else:
            v_r = 20.0 - 10.0 * np.exp(-0.0693 * (t - 30.0))

        # 2. Reference Heading Angle (psi_r)
        psi_r = 0.0

        # 3. Reference Pitch Angle (theta_r)
        if t < 30.0:
            theta_r = 0.0
        elif 30.0 <= t <= 50.0:
            theta_r = 30.0 * (1.0 - np.exp(-0.2 * (t - 30.0)))
        else:
            # Exponential decay after t = 50
            theta_r = 30.0 * (np.exp(-0.2 * (t - 50.0)) - np.exp(-0.2 * (t - 30.0)))

        # Update the internal state and return it
        return v_r, psi_r, theta_r

    # ------------------------------------------------------------------
    # UAV KINEMATICS  (Eq. 1 & 2)
    # ------------------------------------------------------------------

    def _apply_uav_kinematics(self, agent, action, agent_id):
        """
        Apply fixed-wing UAV dynamics from Eq. 1 & 2.
        Disturbances active only during t ∈ [20, 60] s (Section VI).
        action: [v_c, psi_c, theta_c]
        """
        v, psi, theta = self.agent_flight_states[agent]
        v_c, psi_c, theta_c = action

        # --- Disturbance model (Section VI) ---
        in_disturbance_window = 20.0 <= self.t <= 60.0
        varpi = np.random.randn() if in_disturbance_window else 0.0

        if in_disturbance_window:
            d_x = 2.0 * np.sin(self.t)
            d_y = 3.0 * np.cos(self.t)
            d_z = 2.0 * np.sin(self.t)
            d_v     = 2.0 * np.sin(self.t) + varpi
            d_psi   = 2.0 * np.cos(self.t) + varpi
            d_theta = 2.0 * np.sin(self.t) + varpi
        else:
            d_x = d_y = d_z = d_v = d_psi = d_theta = 0.0

        # Eq. 2: autopilot first-order dynamics (no model uncertainty → Δρ = 0)
        v_dot     = self.rho_v     * (v_c     - v)
        psi_dot   = self.rho_psi   * (psi_c   - psi)
        theta_dot = self.rho_theta * (theta_c - theta)

        # Euler integration of flight states + disturbance injection
        v_new     = v     + (v_dot     + d_v)     * self.controller_dt
        psi_new   = psi   + (psi_dot   + d_psi)   * self.controller_dt
        theta_new = theta + (theta_dot + d_theta)  * self.controller_dt

        self.agent_flight_states[agent] = np.array([v_new, psi_new, theta_new])

        psi_rad = np.deg2rad(psi_new)
        theta_rad = np.deg2rad(theta_new)

        # Eq. 1: spatial velocity kinematics
        x_dot = v_new * np.cos(theta_rad) * np.cos(psi_rad) + d_x
        y_dot = v_new * np.cos(theta_rad) * np.sin(psi_rad) + d_y
        z_dot = v_new * np.sin(theta_rad) + d_z

        # Sync orientation visually (Roll=0, Pitch=−θ, Yaw=ψ)
        quat = p.getQuaternionFromEuler([0.0, -theta_rad, psi_rad])
        pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.CLIENT)
        p.resetBasePositionAndOrientation(agent_id, pos, quat, physicsClientId=self.CLIENT)

        # Set velocity in PyBullet (integrates position each stepSimulation)
        p.resetBaseVelocity(
            agent_id,
            linearVelocity=[x_dot, y_dot, z_dot],
            physicsClientId=self.CLIENT
        )

    # ------------------------------------------------------------------
    # SYSTEM MATRICES  (Eq. 11)
    # ------------------------------------------------------------------

    def _compute_FiGi(self, agent):
        """
        Compute nonlinear drift F_i and input matrix G_i from Eq. 11.

        State:  X_i = [e_i(3), Δζ_i(3)] = [ex, ey, ez, Δv, Δψ, Δθ]
        Input:  u_i = [v_c, ψ_c, θ_c]

        F_i (6,):
            [0:3] position error dynamics  = ẋ_i(v,ψ,θ) − ẋ_r(v_r,ψ_r,θ_r) − η̇_ir
            [3:6] flight-state error dynamics = −ρ·ζ_i − ζ̇_r

        G_i (6,3):
            [0:3, :] = 0   (control does not directly appear in position eq.)
            [3:6, :] = diag(ρ_v, ρ_ψ, ρ_θ)
        """
        v,   psi,   theta   = self.agent_flight_states[agent]
        vr,  psi_r, theta_r = self.leader_state

        psi_rad = np.deg2rad(psi)
        theta_rad = np.deg2rad(theta)

        psi_r_rad = np.deg2rad(psi_r)
        theta_r_rad = np.deg2rad(theta_r)

        # η̇_ir = 0 because desired formation offsets are constant in Section VI
        eta_ir_dot = np.zeros(3)

        # Position part of F_i: ẋ_i − ẋ_r − η̇_ir
        xi_dot_agent  = np.array([
            v  * np.cos(theta_rad)   * np.cos(psi_rad),
            v  * np.cos(theta_rad)   * np.sin(psi_rad),
            v  * np.sin(theta_rad)
        ])
        xi_dot_leader = np.array([
            vr * np.cos(theta_r_rad) * np.cos(psi_r_rad),
            vr * np.cos(theta_r_rad) * np.sin(psi_r_rad),
            vr * np.sin(theta_r_rad)
        ])
        Fi_pos = xi_dot_agent - xi_dot_leader - eta_ir_dot

        # Flight-state part of F_i: −ρ·ζ_i − ζ̇_r
        # ζ̇_r = 0 (leader maintains constant v_r, ψ_r, θ_r in Section VI)
        zeta_r_dot = (np.array(self.get_leader_state(self.t)) - np.array(self.get_leader_state(self.t - self.controller_dt))) / self.controller_dt
        Fi_vel = np.array([
            -self.rho_v     * v,
            -self.rho_psi   * psi,
            -self.rho_theta * theta
        ]) - zeta_r_dot

        Fi = np.concatenate([Fi_pos, Fi_vel])   # (6,)

        # G_i: only lower 3×3 block is non-zero (Eq. 11)
        Gi = np.zeros((6, 3))
        Gi[3, 0] = self.rho_v
        Gi[4, 1] = self.rho_psi
        Gi[5, 2] = self.rho_theta

        return Fi, Gi

    # ------------------------------------------------------------------
    # SAFETY BOUND  (Remark 1 / Eq. 10)
    # ------------------------------------------------------------------

    def _compute_k_ie(self, agent):
        pos_i = self.agent_positions[agent]
        min_dist = np.inf

        for other in self.agents:
            if other == agent:
                continue
            dist = np.linalg.norm(pos_i - self.agent_positions[other])
            if dist < min_dist:
                min_dist = dist

        # Formula from Eq 10: k_i <= 1/2 * (||eta_i - eta_j||)
        dynamic_bound = max(min_dist / 2.0, 1e-3)

        ei = np.abs(pos_i - self.eta_ir[agent] - self.leader_pos)

        # Clip to the absolute limits given in Table I
        safe_bound = np.maximum([dynamic_bound] * 3, ei + 5.0)
        k_ie = np.maximum(safe_bound, self.k_e)

        return k_ie

    # ------------------------------------------------------------------
    # KINEMATIC INFORMATION UPDATE
    # ------------------------------------------------------------------

    def _update_kinematic_information(self):
        """Fetch latest positions and velocities from PyBullet."""
        for i, agent in enumerate(self.agents):
            agent_id = self.agent_ids[i]
            pos, _   = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.CLIENT)
            self.agent_positions[agent] = np.array(pos)

            lin_vel, _ = p.getBaseVelocity(agent_id, physicsClientId=self.CLIENT)
            self.agent_velocities[agent] = np.array(lin_vel)

    # ------------------------------------------------------------------
    # OBSERVATION  (Eq. 4)
    # ------------------------------------------------------------------

    def _get_obs(self):
        """
        Return augmented state X_i = [e_i^T, Δζ_i^T]^T for each agent.
            e_i     = η_i − η_ir − η_r   (position error,      shape 3)
            Δζ_i    = ζ_i − ζ_r          (flight-state error,  shape 3)
            X_i     = [e_i, Δζ_i]        (augmented state,     shape 6)
        """
        obs = {}
        for agent in self.agents:
            ei        = (self.agent_positions[agent]
                         - self.eta_ir[agent]
                         - self.leader_pos)                       # (3,)
            delta_zeta = (self.agent_flight_states[agent]
                          - self.leader_state)                    # (3,)
            obs[agent] = np.concatenate([ei, delta_zeta])        # (6,)
        return obs

    # ------------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------------

    def close(self):
        p.disconnect(physicsClientId=self.CLIENT)
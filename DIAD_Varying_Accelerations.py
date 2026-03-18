import time
import numpy as np
import pybullet as p
import pybullet_data


class AffineFormationControl:
    def __init__(self, gui=True):
        super().__init__()

        # Environment parameters
        self.w = 100
        self.h = 100
        self.z = 1.0

        # Options
        self.GUI = gui

        # Connect to PyBullet
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=55,
                cameraYaw=0,
                cameraPitch=-89.99,
                cameraTargetPosition=[50., 50., 1.],
                physicsClientId=self.CLIENT
            )
        else:
            self.CLIENT = p.connect(p.DIRECT)

        # Agent properties
        self.agent_mass = 0.027
        self.agent_radius = 0.06

        # Action Interval
        self.controller_dt = 0.01  # 100Hz Controller

        # Control gains for double integrator (Theorem 8)
        self.k_p = 5.0  # Position gain
        self.k_v = 2.0  # Velocity gain

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None:
            np.random.seed(seed)

        # Reset PyBullet
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems()

        # Get the nominal data (the geometric "blueprint")
        a, b, c = self.get_nominal_formation_data()
        self.nominal_configuration_matrix = a
        self.nominal_augmented_matrix = b
        self.stress_matrix = c
        self.get_partitioned_matrices(self.stress_matrix)

        # Place them randomly
        self.initialize_formation_positions(self.nominal_configuration_matrix)

        # Housekeeping
        self._housekeeping()

        # Initialize kinematic information storage
        self.agent_positions = {}
        self.agent_velocities = {}
        self.agent_accelerations = {}  # NEW: Store accelerations
        self.prev_velocities = {}  # NEW: For computing numerical derivatives

        # Calculate gamma_i for each follower (Proposition 1)
        # γᵢ = Σⱼ∈Nᵢ ωᵢⱼ = [Ω]ᵢᵢ
        self.gamma = {}
        for i, follower in enumerate(self.followers):
            follower_idx = self.num_leaders + i
            self.gamma[follower] = self.stress_matrix[follower_idx, follower_idx]
            print(f"γ_{follower} = {self.gamma[follower]:.6f}")

        # Update kinematic information
        self._update_kinematic_information()

    def get_leader_trajectory(self, leader, t):
        """
        Compute leader velocity and acceleration at time t.
        Returns: (velocity, acceleration)
        """
        # Get leader index
        leader_idx = self.leaders.index(leader)

        # Get nominal position for this leader
        r_i = self.nominal_configuration_matrix[leader_idx, :]

        # Get affine transformation and its derivatives
        A, b, A_dot, b_dot, A_ddot, b_ddot = self.get_affine_transformation(t)

        # Compute desired velocity: v* = A_dot * r_i + b_dot
        vel = A_dot @ r_i + b_dot

        # Compute desired acceleration: a* = A_ddot * r_i + b_ddot
        acc = A_ddot @ r_i + b_ddot

        return vel, acc

    def get_affine_transformation(self, t):
        """
        Updates the affine transformation to include acceleration,
        constant velocity, rotation, and deceleration phases.
        """
        # Initialize variables
        A = np.eye(2)
        A_dot = np.zeros((2, 2))
        A_ddot = np.zeros((2, 2))
        b_dot = np.array([0.0, 0.0])
        b_ddot = np.array([0.0, 0.0])

        # 1. PHASE 1: Straight line movement with Accel/Const Vel (0-20s)
        if t < 20.0:
            max_vel = 1.0  # Target velocity
            accel_time = 5.0  # Time to reach max_vel

            if t < accel_time:
                # Linear Acceleration: v = a*t
                acc_val = max_vel / accel_time
                b_ddot = np.array([0.0, acc_val])
                b_dot = b_ddot * t
            elif t < (20.0 - accel_time):
                # Constant Velocity
                b_dot = np.array([0.0, max_vel])
                b_ddot = np.array([0.0, 0.0])
            else:
                # This segment could handle deceleration, but your prompt
                # implies the full stop is at the end (after 60s).
                b_dot = np.array([0.0, max_vel])
                b_ddot = np.array([0.0, 0.0])

        # 2. PHASE 2: 90 Degree Clockwise Rotation (20-40s)
        elif t < 40.0:
            duration = 20.0
            elapsed = t - 20.0
            total_angle = -np.pi / 2  # 90 degrees clockwise
            omega = total_angle / duration  # Constant angular velocity

            theta = (np.pi / 2) + (omega * elapsed)  # Starting from your initial 90deg offset

            A = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            # A_dot = dA/dt
            A_dot = np.array([
                [-np.sin(theta) * omega, -np.cos(theta) * omega],
                [np.cos(theta) * omega, -np.sin(theta) * omega]
            ])
            # A_ddot = d²A/dt² (Zero if omega is constant)
            A_ddot = np.array([
                [-np.cos(theta) * (omega ** 2), np.sin(theta) * (omega ** 2)],
                [-np.sin(theta) * (omega ** 2), -np.cos(theta) * (omega ** 2)]
            ])

        # 3. PHASE 3: Constant Velocity and Deceleration (40-60s+)
        elif t < 60.0:
            max_vel = 1.0
            decel_start = 55.0  # Start slowing down at 55s

            if t < decel_start:
                b_dot = np.array([max_vel, 0.0])  # Moving right after rotation
                b_ddot = np.array([0.0, 0.0])
            else:
                # Deceleration: v = v0 - a*t
                elapsed_decel = t - decel_start
                decel_val = max_vel / (60.0 - decel_start)
                b_ddot = np.array([-decel_val, 0.0])
                b_dot = np.array([max_vel - (decel_val * elapsed_decel), 0.0])
        else:
            # Full stop
            b_dot = np.array([0.0, 0.0])
            b_ddot = np.array([0.0, 0.0])

        # Numerical integration for position b
        self.prev_b = self.prev_b + b_dot * self.controller_dt
        b = self.prev_b
        self.prev_A = A

        return A, b, A_dot, b_dot, A_ddot, b_ddot

    def get_nominal_formation_data(self):
        """Define the nominal formation configuration."""
        P = np.array([
            [2, 0],  # Agent 1 (Leader)
            [1, 1],  # Agent 2 (Leader)
            [1, -1],  # Agent 3 (Leader)
            [0, 1],  # Agent 4 (Follower)
            [0, -1],  # Agent 5 (Follower)
            [-1, 1],  # Agent 6 (Follower)
            [-1, -1]  # Agent 7 (Follower)
        ])
        self.num_agents = 7
        self.num_leaders = 3
        self.num_followers = 4

        P_bar = np.hstack([P, np.ones((self.num_agents, 1))])

        # Edge weights from paper's example (Fig. 3)
        edges = [
            (0, 1, 0.2741), (0, 2, 0.2741), (0, 3, -0.1370), (0, 4, -0.1370),
            (1, 3, 0.5482), (1, 6, -0.1370),
            (2, 4, 0.5482), (2, 5, -0.1370),
            (3, 4, 0.0685), (3, 5, 0.2741),
            (4, 6, 0.2741),
            (5, 6, 0.1370)
        ]

        # Build stress matrix Omega
        Omega = np.zeros((self.num_agents, self.num_agents))
        for i, j, weight in edges:
            Omega[i, j] = -weight
            Omega[j, i] = -weight

        for i in range(self.num_agents):
            Omega[i, i] = -np.sum(Omega[i, :])

        return P, P_bar, Omega

    def get_partitioned_matrices(self, Omega):
        """Partition the stress matrix according to leaders and followers."""
        n = Omega.shape[0]
        d = 2
        nl = 3
        nf = n - nl

        Omega_bar = np.kron(Omega, np.eye(d))
        cut = nl * d

        self.Omega_ll = Omega_bar[0:cut, 0:cut]
        self.Omega_lf = Omega_bar[0:cut, cut:]
        self.Omega_fl = Omega_bar[cut:, 0:cut]
        self.Omega_ff = Omega_bar[cut:, cut:]

    def initialize_formation_positions(self, P_nominal):
        """Initialize agent positions."""

        self.t = 0.0
        margin = 10
        low, high = margin, self.w - margin

        #center_l = np.random.uniform(low, high, size=2)
        center_l = np.array([10, 10])
        angle = np.pi / 2
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

        self.prev_A = rot_matrix
        self.prev_b = center_l

        self.leader_pos_init = []
        for i in range(self.num_leaders):
            pos = rot_matrix @ P_nominal[i] + center_l
            self.leader_pos_init.append([pos[0], pos[1], self.z])

        self.follower_pos_init = []
        for j in range(self.num_followers):
            i = j + self.num_leaders
            pos = rot_matrix @ P_nominal[i] + center_l
            self.follower_pos_init.append([pos[0], pos[1], self.z])

        self.leaders = [f"leader_{i}" for i in range(self.num_leaders)]
        self.followers = [f"follower_{i}" for i in range(self.num_followers)]
        self.possible_agents = self.leaders + self.followers
        self.agents = self.possible_agents

        self.leader_ids = []
        self.follower_ids = []
        self.agent_ids = []

    def _housekeeping(self):
        """Initialize PyBullet environment and create agents"""
        self.RESET_TIME = time.time()
        self.step_counter = 0

        p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.controller_dt, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        # Create plane
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.01],
            rgbaColor=[1, 1, 1, 1],
        )
        self.PLANE_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plane_vis,
            basePosition=[50, 50, 0]
        )

        # Create leader agents (red)
        for i, agent_name in enumerate(self.leaders):
            agent_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.agent_radius)
            agent_vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.4,
                rgbaColor=[1, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            agent_id = p.createMultiBody(
                baseMass=self.agent_mass,
                baseCollisionShapeIndex=agent_col,
                baseVisualShapeIndex=agent_vis,
                basePosition=[self.leader_pos_init[i][0], self.leader_pos_init[i][1], self.z],
                physicsClientId=self.CLIENT
            )
            p.changeDynamics(
                agent_id, -1,
                ccdSweptSphereRadius=self.agent_radius * 0.9,
                contactProcessingThreshold=0.0,
                physicsClientId=self.CLIENT
            )
            self.leader_ids.append(agent_id)

        # Create follower agents (blue)
        for i, agent_name in enumerate(self.followers):
            agent_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.agent_radius)
            agent_vis = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=0.4,
                rgbaColor=[0, 0, 1, 1],
                physicsClientId=self.CLIENT
            )
            agent_id = p.createMultiBody(
                baseMass=self.agent_mass,
                baseCollisionShapeIndex=agent_col,
                baseVisualShapeIndex=agent_vis,
                basePosition=[self.follower_pos_init[i][0], self.follower_pos_init[i][1], self.z],
                physicsClientId=self.CLIENT
            )
            p.changeDynamics(
                agent_id, -1,
                ccdSweptSphereRadius=self.agent_radius * 0.9,
                contactProcessingThreshold=0.0,
                physicsClientId=self.CLIENT
            )
            self.follower_ids.append(agent_id)

        self.agent_ids = self.leader_ids + self.follower_ids

    def step(self, actions):
        """Execute one simulation step"""
        # Update simulation time
        self.t += self.controller_dt

        clipped_actions = {}
        for agent in self.agents:
            clipped_actions[agent] = self._preprocess_action(agent, actions[agent])

        # Physics simulation
        for agent in self.agents:
            self._apply_physics(agent, clipped_actions[agent])
        p.stepSimulation(physicsClientId=self.CLIENT)

        # Update kinematic information
        self._update_kinematic_information()

        # Advance step counter
        self.step_counter += 1

    def _apply_physics(self, agent, acc):
        """
        Apply acceleration to an agent in PyBullet.
        For double integrator: we apply force F = m*a
        """
        agent_idx = self.possible_agents.index(agent)
        agent_id = self.agent_ids[agent_idx]

        force = self.agent_mass * np.array([acc[0], acc[1], 0])

        p.applyExternalForce(
            objectUniqueId=agent_id,
            linkIndex=-1,
            forceObj=force,
            posObj=self.agent_positions[agent],  # Apply at current position
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT
        )

    def _update_kinematic_information(self):
        """Update positions and velocities for all agents"""
        for i, agent in enumerate(self.possible_agents):
            agent_id = self.agent_ids[i]
            pos, _ = p.getBasePositionAndOrientation(agent_id, physicsClientId=self.CLIENT)
            vel, _ = p.getBaseVelocity(agent_id, physicsClientId=self.CLIENT)

            self.agent_positions[agent] = np.array(pos)
            self.agent_velocities[agent] = np.array(vel)

            current_vel = np.array(vel)

            # Compute acceleration via numerical differentiation
            if agent in self.prev_velocities:
                acc = (current_vel - self.prev_velocities[agent]) / self.controller_dt
                self.agent_accelerations[agent] = acc[:2]  # Only x, y components
            else:
                self.agent_accelerations[agent] = np.zeros(2)

            self.prev_velocities[agent] = current_vel

    def _preprocess_action(self, agent, action=None):
        """
        Compute control input (ACCELERATION) for an agent.

        For leaders: Returns time-varying acceleration from trajectory
        For followers: Implements control law (25) from Theorem 8:
            v̇ᵢ = -(1/γᵢ) Σⱼ∈Nᵢ ωᵢⱼ[kₚ(pᵢ - pⱼ) + kᵥ(vᵢ - vⱼ) - v̇ⱼ]

        This requires absolute acceleration feedback from neighbors.
        """
        # 1. Leaders follow time-varying trajectories
        if agent in self.leaders:
            vel, acc = self.get_leader_trajectory(agent, self.t)
            return acc  # Return ACCELERATION for double integrator

        # 2. Followers use control law (25)
        # Get follower index in stress matrix
        follower_idx = self.num_leaders + self.followers.index(agent)

        # Get follower state
        p_i = self.agent_positions[agent][:2]
        v_i = self.agent_velocities[agent][:2]

        # Calculate control input
        control = np.zeros(2)

        for j in range(self.num_agents):
            if follower_idx == j: continue

            omega_ij = - self.stress_matrix[follower_idx, j]

            if omega_ij != 0:  # j is a neighbor
                # Get neighbor's position and velocity
                if j < self.num_leaders:
                    # Neighbor is a leader
                    neighbor = self.leaders[j]
                else:
                    # Neighbor is a follower
                    neighbor = self.followers[j - self.num_leaders]

                p_j = self.agent_positions[neighbor][:2]
                v_j = self.agent_velocities[neighbor][:2]
                a_j = self.agent_accelerations[neighbor]

                # Control law (25): Σ ωᵢⱼ[kₚ(pᵢ - pⱼ) + kᵥ(vᵢ - vⱼ) - v̇ⱼ]
                control += omega_ij * (
                        self.k_p * (p_i - p_j) +
                        self.k_v * (v_i - v_j) -
                        a_j
                )

        # Apply 1/γᵢ factor
        ui = -(1.0 / self.gamma[agent]) * control

        return ui

    def close(self):
        """Close the PyBullet connection"""
        p.disconnect(physicsClientId=self.CLIENT)
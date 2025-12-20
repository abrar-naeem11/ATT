"""
Multi-Agent RL for Distributed Affine Formation Control
========================================================
Based on Zhao 2018 (IEEE TAC) - Affine Formation Maneuver Control

Architecture:
1. MARL Level (4 Hz): Agents vote on affine parameters
2. Affine Transform: p*_i = A @ r_i + b
3. PID Level (100 Hz): Track affine-generated targets

Novel Contributions:
- Distributed affine parameter voting
- Stress-matrix-weighted aggregation
- Coordination reward for emergent consensus
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pettingzoo import ParallelEnv
from scipy.ndimage import gaussian_filter


class PIDController:
    def __init__(self, kp, ki, kd, max_output, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.dt = dt
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def compute(self, target_pos, current_pos):
        error = target_pos - current_pos
        
        # Proportional
        p_term = self.kp * error
        
        # Integral with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.max_output, self.max_output)
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        # Clip to max velocity
        speed = np.linalg.norm(output)
        if speed > self.max_output:
            output = output / speed * self.max_output
        
        return output


class AffineFormationEnv(ParallelEnv):
    """
    Multi-Agent Affine Formation Control Environment
    
    Key Innovation: Agents vote on affine transformation parameters
    """
    
    def __init__(self,
                 num_agents=3,
                 formation_distance=3.0,
                 initial_target_xy=None,
                 gui=False):
        super().__init__()
        
        # Agent setup
        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Formation parameters
        self.formation_distance = formation_distance
        self.r_nominal = self._define_nominal_formation()
        
        # Compute stress matrix (from Zhao 2018)
        self.stress_matrix = self._compute_stress_matrix()
        self.agent_weights = self._compute_influence_weights()
        
        print(f"Stress-based agent weights: {self.agent_weights}")
        
        # Affine state variables
        self.current_rotation = 0.0
        self.current_scale = 1.0
        self.centroid_velocity = np.zeros(2)
        
        # Previous state for smoothness reward
        self.prev_rotation = 0.0
        self.prev_scale = 1.0
        
        # Environment parameters
        self.w = 100
        self.h = 100
        self.decision_dt = 0.25  # 4 Hz for MARL
        self.physics_dt = 0.01   # 100 Hz for PID
        self.z = 1.0
        self.max_vel = 2.5
        self.num_rays = 9
        self.r_cap = 7.0
        
        # Reward weights
        self.k_target = 5.0
        self.k_formation = 2.0
        self.k_coordination = 0.5
        self.k_obstacle = 0.05
        self.k_smoothness = 0.1
        
        # Penalties
        self.collision_penalty = 20.0
        self.success_reward = 50.0
        
        # Obstacle parameters
        self.num_wall = 0
        self.num_pillars = 0
        self.num_gap_per_row = 5
        
        # PID controllers
        self.pids = {
            agent: PIDController(kp=8.0, ki=2.0, kd=1.5, 
                               max_output=2.5, dt=self.physics_dt)
            for agent in self.possible_agents
        }
        
        # PyBullet setup
        self.GUI = gui
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(
                cameraDistance=55, cameraYaw=0, cameraPitch=-89.99,
                cameraTargetPosition=[50., 50., 1.], physicsClientId=self.CLIENT
            )
        else:
            self.CLIENT = p.connect(p.DIRECT)
        
        self.TARGET_XY_INIT = initial_target_xy
        
        # Storage
        self.agent_ids = []
        self.agent_positions = {}
        self.agent_velocities = {}
        self.target_pos = np.zeros(2)
        self.last_centroid_dist = 0.0
        
        # Observation and action spaces
        self.observation_spaces = {
            agent: self._get_observation_space(agent)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._get_action_space()
            for agent in self.possible_agents
        }
    
    def _define_nominal_formation(self):
        """
        Define r_i - the nominal formation configuration (V-formation)
        This is the template that gets affine-transformed
        """
        r = np.zeros((self.num_agents, 2))
        
        # Leader at origin
        r[0] = [0, 0]
        
        # V-formation
        for i in range(1, self.num_agents):
            row = (i - 1) // 2
            dist = self.formation_distance * (row + 1)
            
            if i % 2 != 0:  # Left wing
                angle = 3 * np.pi / 4
            else:  # Right wing
                angle = -3 * np.pi / 4
            
            r[i] = [dist * np.cos(angle), dist * np.sin(angle)]
        
        return r
    
    def _compute_stress_matrix(self):
        """
        Compute stress matrix (Omega from Zhao 2018)
        Defines structural rigidity of formation
        """
        n = self.num_agents
        Omega = np.zeros((n, n))
        
        if n == 3:
            # V-formation stress weights
            omega_01 = 0.5  # Leader-Left
            omega_02 = 0.5  # Leader-Right
            omega_12 = 0.3  # Wing-Wing
            
            # Construct Laplacian-like matrix
            Omega[0, 0] = omega_01 + omega_02
            Omega[0, 1] = -omega_01
            Omega[0, 2] = -omega_02
            
            Omega[1, 0] = -omega_01
            Omega[1, 1] = omega_01 + omega_12
            Omega[1, 2] = -omega_12
            
            Omega[2, 0] = -omega_02
            Omega[2, 1] = -omega_12
            Omega[2, 2] = omega_02 + omega_12
        
        return Omega
    
    def _compute_influence_weights(self):
        """
        Agent influence based on stress matrix diagonal
        (node strength in formation graph)
        """
        diag = np.diag(self.stress_matrix)
        weights = diag / np.sum(diag)
        return weights
    
    def _get_action_space(self):
        """
        Each agent votes on affine parameters:
        [delta_rotation, delta_scale, delta_vx, delta_vy]
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    def _get_observation_space(self, agent):
        """
        Observation: Own state + Neighbors + Global affine + Obstacles
        Size: 4 + 9 + 4 + 8 = 25
        """
        return spaces.Box(
            low=np.array([
                0, 0, -1, 0,                    # Own state (x, y, angle, speed)
                0, 0, 0, 0, 0, 0, 0, 0, 0,      # 9 rays
                -np.pi, 0.1, -2, -2,            # Global affine (rot, scale, vx, vy)
                0, 0, -1, 0,                    # Neighbor 1
                0, 0, -1, 0,                    # Neighbor 2
                0, 0                            # Target position
            ]),
            high=np.array([
                1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                np.pi, 3.0, 2, 2,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1
            ]),
            dtype=np.float32
        )
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def set_difficulty(self, num_wall, num_gap_per_row, num_pillars):
        """Update obstacle complexity"""
        self.num_wall = num_wall
        self.num_gap_per_row = num_gap_per_row
        self.num_pillars = num_pillars
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset PyBullet
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems()
        
        # Reset affine parameters
        self.current_rotation = 0.0
        self.current_scale = 1.0
        self.centroid_velocity = np.array([0.5, 0.0])  # Initial forward motion
        
        self.prev_rotation = 0.0
        self.prev_scale = 1.0
        
        # Initialize formation centroid
        self.formation_centroid = np.random.uniform([10, 10], [20, 20])
        
        # Initialize target
        if self.TARGET_XY_INIT is None:
            self.target_pos_init = np.random.uniform([5, 94], [95, 96])
        else:
            self.target_pos_init = self.TARGET_XY_INIT
        
        # Setup environment
        self._housekeeping()
        
        # Reset PIDs
        for pid in self.pids.values():
            pid.reset()
        
        # Update kinematics
        self._update_kinematic_information()
        
        # Compute initial observations
        observations = {
            agent: self._compute_obs(agent)
            for agent in self.agents
        }
        
        # Initialize distance tracking
        self.last_centroid_dist = np.linalg.norm(
            self.target_pos[:2] - self.formation_centroid
        )
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """
        Main control loop:
        1. Aggregate votes → global affine parameters
        2. Generate formation targets via affine transform
        3. PID tracks targets at high frequency
        """
        
        # === STEP 1: Aggregate Agent Votes ===
        votes = np.array([actions[agent] for agent in self.agents])
        
        # Stress-weighted average (KEY INNOVATION)
        global_vote = self.agent_weights @ votes
        
        delta_rotation = global_vote[0] * (np.pi / 15)   # ±12°
        delta_scale = global_vote[1] * 0.1               # ±10%
        delta_vx = global_vote[2] * 0.25
        delta_vy = global_vote[3] * 0.25
        
        # === STEP 2: Update Global Affine Parameters ===
        self.current_rotation += delta_rotation
        self.current_scale = np.clip(self.current_scale + delta_scale, 0.5, 2.0)
        self.centroid_velocity[0] += delta_vx
        self.centroid_velocity[1] += delta_vy
        
        # Clip velocity
        speed = np.linalg.norm(self.centroid_velocity)
        if speed > 1.0:
            self.centroid_velocity *= 1.0 / speed
        
        # === STEP 3: Construct Affine Transformation ===
        # A = Scale * Rotation matrix
        c, s = np.cos(self.current_rotation), np.sin(self.current_rotation)
        A = self.current_scale * np.array([[c, -s], [s, c]])
        
        # Update centroid position
        self.formation_centroid += self.centroid_velocity * self.decision_dt
        
        # === STEP 4: Generate Formation Targets (Zhao 2018 Eq 1) ===
        # p*_i = A @ r_i + b
        formation_targets = {}
        for i, agent in enumerate(self.agents):
            formation_targets[agent] = A @ self.r_nominal[i] + self.formation_centroid
        
        # === STEP 5: PID Tracking Loop (100 Hz) ===
        num_substeps = int(self.decision_dt / self.physics_dt)
        
        for _ in range(num_substeps):
            for agent in self.agents:
                current_pos = self.agent_positions[agent][:2]
                target_pos = formation_targets[agent]
                
                # PID computes velocity command
                pid_vel = self.pids[agent].compute(target_pos, current_pos)
                self._apply_physics(agent, pid_vel)
            
            p.stepSimulation(physicsClientId=self.CLIENT)
            self._update_kinematic_information()
        
        # === STEP 6: Compute Rewards ===
        rewards = self._compute_rewards(actions, formation_targets)
        
        # === STEP 7: Check Termination ===
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        
        # === STEP 8: Generate Observations ===
        observations = {
            agent: self._compute_obs(agent)
            for agent in self.agents
        }
        
        infos = {
            agent: {"success": terminated and not truncated}
            for agent in self.agents
        }
        
        # Update state
        self.prev_rotation = self.current_rotation
        self.prev_scale = self.current_scale
        self.step_counter += 1
        
        return observations, rewards, terminations, truncations, infos
    
    def _compute_rewards(self, actions, formation_targets):
        """
        MARL Reward Structure:
        1. Global: Formation reaching target (shared)
        2. Local: Formation tracking error
        3. Coordination: Vote alignment with neighbors
        4. Smoothness: Penalize jerky affine changes
        """
        rewards = {}
        
        # === Global Reward (Shared) ===
        current_centroid_dist = np.linalg.norm(
            self.target_pos[:2] - self.formation_centroid
        )
        r_global = self.k_target * (self.last_centroid_dist - current_centroid_dist)
        self.last_centroid_dist = current_centroid_dist
        
        # Success bonus
        if current_centroid_dist < 2.0:
            r_global += self.success_reward
        
        # === Per-Agent Rewards ===
        for agent in self.agents:
            # 1. Formation tracking error
            current_pos = self.agent_positions[agent][:2]
            target_pos = formation_targets[agent]
            tracking_error = np.linalg.norm(target_pos - current_pos)
            r_tracking = -self.k_formation * tracking_error
            
            # 2. Obstacle avoidance
            obs = self._compute_obs(agent)
            ray_dists = obs[4:13]
            min_ray = np.min(ray_dists)
            r_obs = self.k_obstacle * (min_ray - 0.3)
            
            if min_ray < 0.1:
                r_obs -= self.collision_penalty
            
            # 3. Coordination reward (vote similarity with neighbors)
            r_coord = self._compute_coordination_reward(agent, actions)
            
            # 4. Smoothness reward
            r_smooth = -self.k_smoothness * (
                np.abs(self.current_rotation - self.prev_rotation) +
                np.abs(self.current_scale - self.prev_scale)
            )
            
            # Total reward
            rewards[agent] = float(r_global + r_tracking + r_obs + r_coord + r_smooth)
        
        return rewards
    
    def _compute_coordination_reward(self, agent, actions):
        """
        Reward for voting similar to neighbors
        Encourages emergent consensus
        """
        agent_idx = self.agents.index(agent)
        agent_vote = actions[agent]
        
        # Get neighbors from stress matrix
        neighbors = np.where(self.stress_matrix[agent_idx] < 0)[0]
        
        if len(neighbors) == 0:
            return 0.0
        
        # Compute vote similarity (cosine similarity)
        similarities = []
        for neighbor_idx in neighbors:
            neighbor = self.agents[neighbor_idx]
            neighbor_vote = actions[neighbor]
            
            # Cosine similarity
            dot = np.dot(agent_vote, neighbor_vote)
            norm_product = np.linalg.norm(agent_vote) * np.linalg.norm(neighbor_vote)
            
            if norm_product > 1e-6:
                similarity = dot / norm_product
                similarities.append(similarity)
        
        return self.k_coordination * np.mean(similarities) if similarities else 0.0
    
    def _compute_obs(self, agent):
        """Compute observation for agent"""
        agent_idx = self.agents.index(agent)
        
        # Own state
        pos = self.agent_positions[agent][:2]
        vel = self.agent_velocities[agent][:2]
        angle = np.arctan2(vel[1], vel[0])
        speed = np.linalg.norm(vel)
        
        obs = np.zeros(25)
        
        # [0-3] Own state
        obs[0] = pos[0] / self.w
        obs[1] = pos[1] / self.h
        obs[2] = angle / np.pi
        obs[3] = speed / self.max_vel
        
        # [4-12] Ray distances
        ray_dists = self._compute_ray_distances(pos[0], pos[1], angle)
        obs[4:13] = ray_dists
        
        # [13-16] Global affine parameters
        obs[13] = self.current_rotation / np.pi
        obs[14] = self.current_scale / 3.0
        obs[15] = self.centroid_velocity[0] / 2.0
        obs[16] = self.centroid_velocity[1] / 2.0
        
        # [17-24] Neighbor states
        neighbors = np.where(self.stress_matrix[agent_idx] < 0)[0]
        neighbor_obs = []
        
        for neighbor_idx in neighbors:
            neighbor = self.agents[neighbor_idx]
            n_pos = self.agent_positions[neighbor][:2]
            n_vel = self.agent_velocities[neighbor][:2]
            n_angle = np.arctan2(n_vel[1], n_vel[0])
            n_speed = np.linalg.norm(n_vel)
            
            neighbor_obs.extend([
                n_pos[0] / self.w,
                n_pos[1] / self.h,
                n_angle / np.pi,
                n_speed / self.max_vel
            ])
        
        # Pad if fewer than 2 neighbors
        while len(neighbor_obs) < 8:
            neighbor_obs.extend([0, 0, 0, 0])
        
        obs[17:25] = neighbor_obs[:8]
        
        return obs
    
    def get_global_affine_state(self):
        """Return current global affine parameters"""
        return np.array([
            self.current_rotation,
            self.current_scale,
            self.centroid_velocity[0],
            self.centroid_velocity[1]
        ])
    
    def _compute_terminated(self):
        """Check if target reached"""
        dist = np.linalg.norm(self.target_pos[:2] - self.formation_centroid)
        return dist < 2.0
    
    def _compute_truncated(self):
        """Check for collisions or timeout"""
        # Check collisions
        for agent in self.agents:
            pos = self.agent_positions[agent][:2]
            vel = self.agent_velocities[agent][:2]
            angle = np.arctan2(vel[1], vel[0])
            ray_dists = self._compute_ray_distances(pos[0], pos[1], angle)
            
            if np.any(ray_dists < 0.1):
                return True
        
        # Check timeout
        if self.step_counter > 1000:
            return True
        
        return False
    
    def _housekeeping(self):
        """Initialize PyBullet environment"""
        self.step_counter = 0
        self.agent_ids = []
        self.agent_positions = {}
        self.agent_velocities = {}
        
        # PyBullet setup
        p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.physics_dt, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        
        # Create plane
        plane_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[50, 50, 0.01], rgbaColor=[1, 1, 1, 1]
        )
        p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plane_vis, basePosition=[50, 50, 0]
        )
        
        # Create agents at initial formation positions
        A_init = np.eye(2)  # Identity transform initially
        for i, agent in enumerate(self.agents):
            init_pos = A_init @ self.r_nominal[i] + self.formation_centroid
            
            agent_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.06)
            agent_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.4, rgbaColor=[0, 0, 1, 1]
            )
            agent_id = p.createMultiBody(
                baseMass=0.027, baseCollisionShapeIndex=agent_col,
                baseVisualShapeIndex=agent_vis,
                basePosition=[init_pos[0], init_pos[1], self.z]
            )
            self.agent_ids.append(agent_id)
        
        # Create target
        target_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.4, rgbaColor=[1, 0, 0, 1]
        )
        self.TARGET_ID = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_vis,
            basePosition=[self.target_pos_init[0], self.target_pos_init[1], self.z]
        )
        
        # Add obstacles
        self._add_obstacles()
    
    def _add_obstacles(self):
        """Add obstacles to environment"""
        self.obstacle_ids = []
        
        # Walls with gaps
        for i in range(self.num_wall):
            y = 10 + i * 20
            gap_centers = np.sort(np.random.uniform(5, 95, self.num_gap_per_row))
            gap_width = 3.0
            
            current_x = 0
            for center in gap_centers:
                gap_start = center - gap_width / 2
                if gap_start > current_x:
                    seg_width = gap_start - current_x
                    pos_x = current_x + seg_width / 2
                    self._spawn_box([pos_x, y, 1], [seg_width/2, 1, 2])
                current_x = center + gap_width / 2
            
            if current_x < 100:
                seg_width = 100 - current_x
                self._spawn_box([current_x + seg_width/2, y, 1], [seg_width/2, 1, 2])
        
        # Random pillars
        placed = 0
        while placed < self.num_pillars:
            px, py = np.random.uniform(10, 90), np.random.uniform(15, 90)
            too_close = any(
                np.linalg.norm([px, py] - p.getBasePositionAndOrientation(oid)[0][:2]) < 8
                for oid in self.obstacle_ids
            )
            if not too_close:
                self._spawn_box([px, py, 1], [1.2, 1.2, 2])
                placed += 1
        
        # Boundary walls
        self._spawn_box([50, 0, 1], [50, 0.1, 2])
        self._spawn_box([50, 100, 1], [50, 0.1, 2])
        self._spawn_box([0, 50, 1], [0.1, 50, 2])
        self._spawn_box([100, 50, 1], [0.1, 50, 2])
    
    def _spawn_box(self, pos, half_extents):
        """Spawn obstacle box"""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, 
                                  rgbaColor=[0.6, 0.6, 0.6, 1])
        oid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                               baseVisualShapeIndex=vis, basePosition=pos)
        self.obstacle_ids.append(oid)
    
    def _apply_physics(self, agent, vel):
        """Apply velocity to agent"""
        agent_idx = self.agents.index(agent)
        agent_id = self.agent_ids[agent_idx]
        p.resetBaseVelocity(agent_id, linearVelocity=[vel[0], vel[1], 0],
                          angularVelocity=[0, 0, 0], physicsClientId=self.CLIENT)
    
    def _update_kinematic_information(self):
        """Update agent positions and velocities"""
        for i, agent in enumerate(self.agents):
            pos, _ = p.getBasePositionAndOrientation(self.agent_ids[i], 
                                                     physicsClientId=self.CLIENT)
            vel, _ = p.getBaseVelocity(self.agent_ids[i], physicsClientId=self.CLIENT)
            self.agent_positions[agent] = np.array(pos)
            self.agent_velocities[agent] = np.array(vel)
        
        self.target_pos, _ = p.getBasePositionAndOrientation(
            self.TARGET_ID, physicsClientId=self.CLIENT
        )
        self.target_pos = np.array(self.target_pos)
    
    def _compute_ray_distances(self, x, y, angle):
        """Compute ray distances for obstacle detection"""
        ray_from, ray_to = [], []
        start_angle = angle - np.pi / 3
        
        for r in range(self.num_rays):
            a = start_angle + r * (np.pi / 12)
            dx, dy = np.cos(a), np.sin(a)
            ray_from.append([x + 0.07*dx, y + 0.07*dy, self.z])
            ray_to.append([x + (self.r_cap+0.07)*dx, y + (self.r_cap+0.07)*dy, self.z])
        
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
        return np.array([r[2] for r in results])
    
    def close(self):
        """Close environment"""
        p.disconnect(physicsClientId=self.CLIENT)

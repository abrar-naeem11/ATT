import numpy as np
import pybullet as p
import pybullet_data
import time


class AffineFormationControl:
    def __init__(self):
        self.w = 100
        self.h = 50
        self.decision_dt = 0.25  # 4 Hz Control Loop
        self.physics_dt = 0.01  # 100 Hz Physics
        self.z = 1.0
        self.leader_vel_max = 5.0
        self.follower_vel_max = 10.0

        # Initialize PyBullet
        self.CLIENT = p.connect(p.GUI)
        for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                  p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS]:
            p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)

        p.resetDebugVisualizerCamera(
            cameraDistance=60,
            cameraYaw=0,
            cameraPitch=-89.99,
            cameraTargetPosition=[25., 0., 1.],
            physicsClientId=self.CLIENT
        )

        # Agent properties
        self.agent_mass = 0.027
        self.agent_radius = 0.3

        # Nominal formation configuration (from Figure 3 in paper)
        # 7 agents in 2D plane - using exact coordinates from the figure
        self.r = self.get_nominal_formation()

        self.n = 7  # Total number of agents
        self.n_l = 3  # Number of leaders (agents 0, 1, 2)
        self.n_f = 4  # Number of followers (agents 3, 4, 5, 6)
        self.d = 2  # Dimension

        # Calculate stress matrix from paper Figure 3
        self.stress_data = self.get_stress_data()
        
        # Build list of edges for visualization
        self.edges = [(i, j) for i, j, w in self.stress_data]
        self.edge_weights = {(i, j): w for i, j, w in self.stress_data}
        
        self.Omega = self.build_stress_matrix()

        # Partition stress matrix for leaders and followers
        self.Omega_ff = self.Omega[self.n_l:, self.n_l:]
        self.Omega_fl = self.Omega[self.n_l:, :self.n_l]

        # Control gains (from paper simulation: k_p = 0.5, k_v = 2.0)
        self.k_p = 0.5
        self.k_v = 2.0

        # Initialize agent states
        self.positions = np.copy(self.r)  # Current positions
        self.velocities = np.zeros_like(self.positions)  # Current velocities

        # Leader trajectory parameters
        self.time = 0.0
        
        # For acceleration computation (numerical differentiation)
        self.prev_leader_pos = None
        self.prev_leader_vel = None

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset PyBullet
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems()

        # Reset time and states
        self.time = 0.0
        self.positions = np.copy(self.r)
        self.velocities = np.zeros_like(self.positions)
        self.prev_leader_pos = None
        self.prev_leader_vel = None

        self._housekeeping()

    def _housekeeping(self):
        self.agent_ids = []

        p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.physics_dt, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        # Create plane
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 25, 0.01],
            rgbaColor=[0.9, 0.9, 0.9, 1],
        )
        self.PLANE_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plane_vis,
            basePosition=[50, 0, 0]
        )

        # Create visual representations for agents
        self.create_agents()

        # Create obstacles matching the paper's simulation
        self.create_obstacles()

    def get_nominal_formation(self):
        """Nominal formation from Figure 3 of the paper"""
        # Configuration with agents positioned as shown in Figure 3
        # Leaders: 0, 1, 2 (red circles)
        # Followers: 3, 4, 5, 6 (blue circles)
        r = np.array([
            [2.0, 0.0],    # Agent 0 (leader) - rightmost
            [1.0, 1.0],    # Agent 1 (leader) - top
            [1.0, -1.0],   # Agent 2 (leader) - bottom
            [0.0, 1.0],    # Agent 3 (follower) - top middle
            [0.0, -1.0],   # Agent 4 (follower) - bottom middle
            [-1.0, 1.0],   # Agent 5 (follower) - top left
            [-1.0, -1.0]   # Agent 6 (follower) - bottom left
        ])
        return r

    def get_stress_data(self):
        """Stress matrix weights from Figure 3 of the paper"""
        # (agent_i, agent_j, weight)
        # Normalized weights as shown in the figure
        stress_data = [
            # Positive weights (attractive forces - green in figure)
            (0, 1, 0.2741), (0, 2, 0.2741),
            (1, 3, 0.5482), (2, 4, 0.5482),
            (3, 4, 0.0685),
            (3, 5, 0.2741), (4, 6, 0.2741),
            (5, 6, 0.1370),
            # Negative weights (repulsive forces - red in figure)
            (1, 6, -0.1370), (2, 5, -0.1370),
            (0, 3, -0.1370), (0, 4, -0.1370)
        ]
        return stress_data

    def build_stress_matrix(self):
        """Build stress matrix Omega"""
        Omega = np.zeros((self.n, self.n))
        for i, j, w in self.stress_data:
            # Off-diagonal elements are -omega_ij
            Omega[i, j] = -w
            Omega[j, i] = -w
            # Diagonal elements sum the weights
            Omega[i, i] += w
            Omega[j, j] += w
        return Omega

    def create_agents(self):
        """Create visual representations of agents in PyBullet"""
        for i in range(self.n):
            # Red for leaders, blue for followers
            color = [1, 0, 0, 1] if i < self.n_l else [0, 0, 1, 1]
            radius = 0.4

            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=radius, 
                rgbaColor=color,
                physicsClientId=self.CLIENT
            )

            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=self.agent_radius,
            )

            agent_id = p.createMultiBody(
                baseMass=self.agent_mass,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=[self.positions[i, 0], self.positions[i, 1], self.z],
                physicsClientId=self.CLIENT
            )
            p.changeDynamics(
                agent_id, -1,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self.CLIENT
            )

            self.agent_ids.append(agent_id)

    def create_obstacles(self):
        """Create obstacles similar to the paper's simulation example"""
        # Create narrow passages/walls to navigate through
        # Based on the trajectory shown in the paper's Figure 4
        
        # Wall 1 - around x=20-25
        self._create_wall_with_gap(x_center=22.5, y_gap_center=0, gap_width=8.0, wall_height=25, wall_thickness=5.0)
        
        # Wall 2 - around x=40-45  
        self._create_wall_with_gap(x_center=42.5, y_gap_center=0, gap_width=6.0, wall_height=25, wall_thickness=5.0)

    def _create_wall_with_gap(self, x_center, y_gap_center, gap_width, wall_height, wall_thickness):
        """Create a wall with a gap in the middle"""
        half_gap = gap_width / 2.0
        
        # Top wall section
        top_height = (wall_height - gap_width) / 2.0
        top_y = y_gap_center + half_gap + top_height / 2.0
        self._create_box(
            pos=[x_center, top_y, self.z],
            half_extents=[wall_thickness/2, top_height/2, 2.0],
            color=[0.4, 0.4, 0.4, 1]
        )
        
        # Bottom wall section
        bottom_y = y_gap_center - half_gap - top_height / 2.0
        self._create_box(
            pos=[x_center, bottom_y, self.z],
            half_extents=[wall_thickness/2, top_height/2, 2.0],
            color=[0.4, 0.4, 0.4, 1]
        )

    def _create_box(self, pos, half_extents, color):
        """Helper to create a box obstacle"""
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color
        )

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )

        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=pos
        )

    def generate_leader_trajectory(self, t):
        """
        Generate time-varying leader positions following affine transformations
        Similar to the trajectory shown in Figure 4 of the paper
        """
        r_leaders = self.r[:self.n_l]  # Nominal leader positions

        # Design trajectory segments to match paper's simulation
        # The formation moves forward, rotates, scales down for narrow passages,
        # then scales back up
        
        if t < 20.0:
            # Phase 1: Move forward with slight rotation
            scale = 1.0
            angle = t * 0.02  # Gradual rotation
            tx = t * 1.0  # Move forward
            ty = 0.0
            
        elif t < 40.0:
            # Phase 2: Approach first obstacle, start scaling down
            dt = t - 20.0
            scale = 1.0 - dt * 0.015  # Scale down to 0.7
            angle = 0.4 + dt * 0.01
            tx = 20.0 + dt * 0.8
            ty = 0.0
            
        elif t < 60.0:
            # Phase 3: Pass through first narrow passage (scaled down)
            dt = t - 40.0
            scale = 0.7
            angle = 0.6 + dt * 0.015
            tx = 36.0 + dt * 0.8
            ty = 0.0
            
        elif t < 80.0:
            # Phase 4: Between obstacles, start scaling back up and rotating
            dt = t - 60.0
            scale = 0.7 + dt * 0.015  # Scale back up to 1.0
            angle = 0.9 + dt * 0.03
            tx = 52.0 + dt * 0.6
            ty = np.sin(dt * 0.1) * 2.0  # Add some vertical movement
            
        elif t < 100.0:
            # Phase 5: Approach second obstacle with different orientation
            dt = t - 80.0
            scale = 1.0 - dt * 0.01  # Scale down slightly
            angle = 1.5 + dt * 0.02
            tx = 64.0 + dt * 0.5
            ty = 2.0 * np.cos(dt * 0.1)
            
        else:
            # Phase 6: Continue forward
            dt = t - 100.0
            scale = 0.8
            angle = 1.9
            tx = 74.0 + dt * 0.4
            ty = 0.0

        # Build affine transformation matrix A
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        A = scale * np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])

        b = np.array([tx, ty])

        # Apply affine transformation: p = A*r + b
        p_leaders = np.array([A @ r_leaders[i] + b for i in range(self.n_l)])

        # Compute leader velocities using numerical differentiation
        if self.prev_leader_pos is not None:
            v_leaders = (p_leaders - self.prev_leader_pos) / self.physics_dt
        else:
            v_leaders = np.zeros_like(p_leaders)

        # Compute leader accelerations using numerical differentiation
        if self.prev_leader_vel is not None:
            a_leaders = (v_leaders - self.prev_leader_vel) / self.physics_dt
        else:
            a_leaders = np.zeros_like(v_leaders)

        # Store for next iteration
        self.prev_leader_pos = p_leaders.copy()
        self.prev_leader_vel = v_leaders.copy()

        return p_leaders, v_leaders, a_leaders

    def control_law(self, p_leaders, v_leaders, a_leaders):
        """
        Control law (25) from the paper for double-integrator dynamics:
        
        dot_v_i = -1/gamma_i * sum_{j in N_i} omega_ij [k_p(p_i - p_j) + k_v(v_i - v_j) - dot_v_j]
        
        where gamma_i = sum_{j in N_i} omega_ij (diagonal element of stress matrix)
        """
        accelerations = np.zeros((self.n_f, self.d))

        for i in range(self.n_f):
            agent_idx = i + self.n_l  # Global index of follower
            gamma_i = self.Omega[agent_idx, agent_idx]  # Diagonal element
            
            if abs(gamma_i) < 1e-6:
                continue  # Skip if gamma_i is zero (shouldn't happen with valid formation)
            
            control_sum = np.zeros(self.d)

            # Sum over all neighbors (both leaders and followers)
            for j in range(self.n):
                if agent_idx == j:
                    continue
                
                # Get stress weight (Omega stores -omega_ij in off-diagonals)
                omega_ij = -self.Omega[agent_idx, j]
                
                if abs(omega_ij) > 1e-6:  # Only compute if edge exists
                    if j < self.n_l:
                        # Neighbor is a leader - use prescribed trajectory
                        p_diff = self.positions[agent_idx] - p_leaders[j]
                        v_diff = self.velocities[agent_idx] - v_leaders[j]
                        a_j = a_leaders[j]
                    else:
                        # Neighbor is a follower - use current state
                        p_diff = self.positions[agent_idx] - self.positions[j]
                        v_diff = self.velocities[agent_idx] - self.velocities[j]
                        a_j = np.zeros(self.d)  # Follower acceleration not in feedback

                    control_sum += omega_ij * (
                        self.k_p * p_diff + 
                        self.k_v * v_diff - 
                        a_j
                    )

            # Apply control law
            accelerations[i] = -control_sum / gamma_i

        return accelerations

    def update(self):
        """Update one simulation step"""
        # Generate leader trajectory (affine transformation of nominal formation)
        p_leaders, v_leaders, a_leaders = self.generate_leader_trajectory(self.time)

        # Update leader states directly (leaders follow prescribed trajectory)
        self.positions[:self.n_l] = p_leaders
        self.velocities[:self.n_l] = v_leaders

        # Compute control for followers using Equation (25)
        accelerations_f = self.control_law(p_leaders, v_leaders, a_leaders)

        # Update follower states (double integrator dynamics)
        # dot_v = a, dot_p = v
        self.velocities[self.n_l:] += accelerations_f * self.physics_dt
        self.positions[self.n_l:] += self.velocities[self.n_l:] * self.physics_dt

        # Update visual positions in PyBullet
        for i, agent_id in enumerate(self.agent_ids):
            p.resetBasePositionAndOrientation(
                agent_id,
                [self.positions[i, 0], self.positions[i, 1], self.z],
                [0, 0, 0, 1],
                physicsClientId=self.CLIENT
            )
            # Set velocities for smooth visualization
            p.resetBaseVelocity(
                agent_id,
                linearVelocity=[self.velocities[i, 0], self.velocities[i, 1], 0],
                physicsClientId=self.CLIENT
            )

        # Visualize formation edges
        self.draw_formation_edges()

        self.time += self.physics_dt

    def draw_formation_edges(self):
        """Draw lines between connected agents to visualize formation structure"""
        p.removeAllUserDebugItems(physicsClientId=self.CLIENT)
        
        for edge in self.edges:
            i, j = edge
            weight = self.edge_weights[edge]
            
            # Green for positive (attractive), red for negative (repulsive)
            if weight > 0:
                color = [0, 1, 0]  # Green
                width = 2.0
            else:
                color = [1, 0, 0]  # Red
                width = 2.0
            
            p.addUserDebugLine(
                [self.positions[i, 0], self.positions[i, 1], self.z],
                [self.positions[j, 0], self.positions[j, 1], self.z],
                lineColorRGB=color,
                lineWidth=width,
                physicsClientId=self.CLIENT
            )

    def run(self, duration=130.0):
        """Run simulation for specified duration"""
        print("Starting Affine Formation Control Simulation")
        print(f"Duration: {duration} seconds")
        print("Red agents: Leaders (following prescribed affine trajectory)")
        print("Blue agents: Followers (using control law from Equation 25)")
        print("Green lines: Attractive forces (positive stress)")
        print("Red lines: Repulsive forces (negative stress)")
        print("\nPress Ctrl+C to stop early\n")
        
        try:
            while self.time < duration:
                self.update()
                p.stepSimulation(physicsClientId=self.CLIENT)
                time.sleep(self.physics_dt)  # Real-time simulation
                
                # Print progress every 10 seconds
                if int(self.time) % 10 == 0 and self.time > 0:
                    if int(self.time) == int(self.time - self.physics_dt) + 1:
                        # Compute tracking error
                        tracking_error = np.linalg.norm(
                            self.velocities[self.n_l:] - 
                            np.mean(self.velocities[:self.n_l], axis=0)
                        )
                        print(f"Time: {int(self.time)}s | Formation centroid: "
                              f"({np.mean(self.positions[:, 0]):.1f}, "
                              f"{np.mean(self.positions[:, 1]):.1f})")

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        print("\nSimulation complete!")
        print(f"Final time: {self.time:.2f}s")
        print("Press Enter to close...")
        input()
        p.disconnect(physicsClientId=self.CLIENT)


if __name__ == "__main__":
    controller = AffineFormationControl()
    controller.run(duration=130.0)

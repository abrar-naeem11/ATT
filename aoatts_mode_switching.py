import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from enums import ActionType, ObservationType
import cvxpy as cp
from typing import Tuple, Optional

class AOATTS(gym.Env):
    """Autonomous Obstacle Avoidance and Target Tracking with Mode-Switching CLF/CBF"""

    def __init__(self,
                 initial_agent_xy=None,
                 initial_target_xy=None,
                 gui=False,
                 output_folder='results',
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL,
                 rl_frequency=1.0,
                 qp_frequency=10.0,
                 physics_frequency=100.0,
                 lambda_clf: float = 1.0,
                 gamma_clf: float = 0.5,
                 alpha_cbf: float = 1.0,
                 distance_threshold: float = 10.0,
                 safety_distance: float = 2.0,
                 cbf_activation_distance: float = 4.0,
                 epsilon: float = 1e-6,
                 mode: str = 'switching'  # 'switching', 'unified', 'clf_only', 'cbf_only'
                 ):
        self.EPISODE_LEN_SEC = 300

        #
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #

        # Observation Parameters
        self.w = 100
        self.h = 100
        self.agent_vel_max = 2
        self.num_rays = 9
        self.r_cap = 15

        # Action Parameters
        self.del_vel_max = 0.5
        self.del_angle_max = np.pi / 6

        # Reward Parameters
        self.k_D = 0.05
        self.k_theta = 0.05
        self.k_obs = 0.01
        self.k_v = 0.01
        self.v_c = 0.5

        # Penalty Parameters
        self.collision_penalty = 10.0
        self.success_penalty = 10.0

        # Obstacle Parameters
        self.num_static_obstacles = 40
        self.num_dynamic_obstacles = 10
        self.obstacle_radius_min = 0.5
        self.obstacle_radius_max = 1.0
        self.dynamic_vel_max = 2.0
        self.dynamic_d_thr = 2

        # Control frequencies
        self.rl_frequency = rl_frequency
        self.qp_frequency = qp_frequency
        self.physics_frequency = physics_frequency

        # Time steps
        self.rl_dt = 1.0 / self.rl_frequency
        self.qp_dt = 1.0 / self.qp_frequency
        self.physics_dt = 1.0 / self.physics_frequency

        # Substeps
        self.physics_steps_per_qp = int(self.qp_dt / self.physics_dt)
        self.qp_steps_per_rl = int(self.rl_dt / self.qp_dt)

        # CLF-CBF QP Parameters
        self.lambda_clf = lambda_clf
        self.gamma_clf = gamma_clf
        self.alpha_cbf = alpha_cbf
        self.distance_threshold = distance_threshold
        self.safety_distance = safety_distance
        self.cbf_activation_distance = cbf_activation_distance
        self.epsilon = epsilon
        self.filter_mode = mode  # 'switching', 'unified', 'clf_only', 'cbf_only'
        
        self.last_rl_action = None
        self.qp_call_count = 0
        self.qp_success_count = 0
        self.cbf_mode_count = 0
        self.clf_mode_count = 0
        self.total_actions = 0.0
        self.filter_info = None

        # Constants#
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180

        # Options#
        self.GUI = gui
        self.OUTPUT_FOLDER = output_folder

        # Connect to PyBullet#
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=50,
                                         cameraYaw=0,
                                         cameraPitch=-89.99,
                                         cameraTargetPosition=[50., 50., 1.],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
        else:
            self.CLIENT = p.connect(p.DIRECT)

        self.AGENT_XY_INIT = initial_agent_xy
        self.TARGET_XY_INIT = initial_target_xy

        self.z = 1.0

        # Create action and observation spaces#
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        self.truncation_reason = "none"
        self.AGENT_TARGET_DIST = 0.
        self.LAST_AGENT_TARGET_DIST = 0.
        self.LAST_ALPHA = 0.
        self.prev_ray_dist = np.zeros(9)

        self.termination_cause = None


    def reset(self,
              seed: int = None,
              options: dict = None,
              ):
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems()

        #Agent#
        if self.AGENT_XY_INIT is None:
            self.INIT_AGENT_XY = np.random.uniform(low=[1.0, 1.0], high=[self.w - 1, self.h - 1])
        else:
            self.INIT_AGENT_XY = self.AGENT_XY_INIT

        #Target#
        if self.TARGET_XY_INIT is None:
            self.INIT_TARGET_XY = np.random.uniform(low=[1.0, 1.0], high=[self.w - 1, self.h - 1])
        else:
            self.INIT_TARGET_XY = self.TARGET_XY_INIT

        #HouseKeeping#
        self._housekeeping()

        #Update and Store the drones kinematic information#
        self._updateAndStoreKinematicInformation()

        self.last_rl_action = None
        self.qp_call_count = 0
        self.qp_success_count = 0
        self.cbf_mode_count = 0
        self.clf_mode_count = 0
        self.total_action_deviation = 0.0

        #Return the initial observation#
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info


    def step(self,
             action
             ):
        #Save, preprocess, and clip the action to max. RPM#
        clipped_action = self._preprocessAction(action)
        self.last_rl_action = clipped_action

        self.LAST_AGENT_TARGET_DIST = np.linalg.norm(
            [self.agent_pos[0] - self.target_pos[0], self.agent_pos[1] - self.target_pos[1]]
        )

        prev_obs = self._computeObs()
        self.prev_ray_dist = prev_obs[6:15]
        self.LAST_ALPHA = prev_obs[17]

        terminated = False
        truncated = False

        for _ in range(self.qp_steps_per_rl):
            filtered_action, qp_info = self.adaptive_filter(self.last_rl_action)
            self.filter_info = qp_info
            self.qp_call_count += 1
            if qp_info['qp_success']:
                self.qp_success_count += 1
            if qp_info['active_mode'] == 'CBF':
                self.cbf_mode_count += 1
            elif qp_info['active_mode'] == 'CLF':
                self.clf_mode_count += 1
            self.total_action_deviation += qp_info['action_deviation']

            for _ in range(self.physics_steps_per_qp):
                self._physics(filtered_action)
                p.stepSimulation(physicsClientId=self.CLIENT)

            #Update and store the drones kinematic information#
            self._updateAndStoreKinematicInformation()

            if self._computeTerminated():
                terminated = True
                break

            if self._computeTruncated():
                truncated = True
                break

        #Store Current Distance between agent and target#
        self.AGENT_TARGET_DIST = np.linalg.norm([self.agent_pos[0]-self.target_pos[0], self.agent_pos[1]-self.target_pos[1]])

        # Advance the step counter#
        self.step_counter = self.step_counter + 1

        #Prepare the return values#
        obs = self._computeObs()
        reward = self._computeReward()
        info = self._computeInfo(terminated, truncated)

        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(physicsClientId=self.CLIENT)

    def getPyBulletClient(self):
        return self.CLIENT

    def getAgentId(self):
        return self.AGENT_ID

    def getTargetId(self):
        return self.TARGET_ID

    def _physics(self,
                 vel
                 ):

        p.resetBaseVelocity(
            self.AGENT_ID,
            linearVelocity=[vel[0], vel[1], 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.CLIENT
        )

        timestep = 0.01
        # Update dynamic obstacles
        for oid in self.dynamic_obs_ids:
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=self.CLIENT)
            vel, _ = p.getBaseVelocity(oid, physicsClientId=self.CLIENT)
            vel = np.array(vel)

            next_pos = np.array(pos) + vel * timestep
            if next_pos[0] < 0 or next_pos[0] > self.w:
                vel[0] = -vel[0]
            if next_pos[1] < 0 or next_pos[1] > self.h:
                vel[1] = -vel[1]

            #Obstacle avoidance check
            vel_angle = np.arctan2(vel[1], vel[0])
            ray_dist_norm = self._computeRayDistances(pos[0], pos[1], vel_angle)
            ray_dists = ray_dist_norm * self.r_cap
            forward_dist = ray_dists[4]

            if forward_dist < self.dynamic_d_thr:
                if vel[0] > 0:
                    vel[0] = np.random.uniform(-self.dynamic_vel_max, -0.5)
                else:
                    vel[0] = np.random.uniform(0.5, self.dynamic_vel_max)
                if vel[1] > 0:
                    vel[1] = np.random.uniform(-self.dynamic_vel_max, -0.5)
                else:
                    vel[1] = np.random.uniform(0.5, self.dynamic_vel_max)

            if np.linalg.norm(vel[:2]) < 0.1:
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(0.5, self.dynamic_vel_max)
                vel[0] = speed * np.cos(angle)
                vel[1] = speed * np.sin(angle)

            p.resetBaseVelocity(oid, vel.tolist(), physicsClientId=self.CLIENT)


    def _housekeeping(self):
        """Housekeeping function."""
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True

        #### Initialize the drones kinemaatic information ##########
        # Agent#
        self.agent_pos = np.zeros(3)
        self.agent_vel = np.zeros(3)

        # Target#
        self.target_pos = np.zeros(3)
        self.target_vel = np.zeros(3)

        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, 0, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(0.01, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        
        plane_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[50, 50, 0.01],
            rgbaColor=[1, 1, 1, 1]
        )

        self.PLANE_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=plane_vis,
            basePosition=[50, 50, 0]
        )

        # Agent
        self.agent_mass = 0.027
        self.agent_radius = 0.06
        agent_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.agent_radius)
        agent_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 0, 1, 1], physicsClientId=self.CLIENT)
        self.AGENT_ID = p.createMultiBody(
            baseMass=self.agent_mass,
            baseCollisionShapeIndex=agent_col,
            baseVisualShapeIndex=agent_vis,
            basePosition=[self.INIT_AGENT_XY[0], self.INIT_AGENT_XY[1], self.z],
            physicsClientId=self.CLIENT
        )
        p.changeDynamics(self.AGENT_ID, -1,
                         ccdSweptSphereRadius=self.agent_radius * 0.9,
                         contactProcessingThreshold=0.0,
                         physicsClientId=self.CLIENT)

        #Target
        target_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT)
        self.TARGET_ID = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_vis,
            basePosition=[self.INIT_TARGET_XY[0], self.INIT_TARGET_XY[1], self.z],
            physicsClientId=self.CLIENT
        )

        self._addObstacles()


    def _addObstacles(self):

        self.obstacle_ids = []
        self.dynamic_obs_ids = []

        total_obstacles = self.num_static_obstacles + self.num_dynamic_obstacles
        for i in range(total_obstacles):
            radius = np.random.uniform(self.obstacle_radius_min, self.obstacle_radius_max)
            pos = np.random.uniform(1, self.w - 1, 2).tolist() + [self.z]
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 0, 1])
            mass = 0 if i < self.num_static_obstacles else 1.0
            oid = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=pos,
                physicsClientId=self.CLIENT
            )
            if mass > 0:
                p.changeDynamics(oid, -1,
                                 ccdSweptSphereRadius=radius * 0.9,
                                 contactProcessingThreshold=0.0,
                                 physicsClientId=self.CLIENT)
            self.obstacle_ids.append(oid)
            if mass > 0:
                self.dynamic_obs_ids.append(oid)
                vel_mag = np.random.uniform(0.5, self.dynamic_vel_max)
                vel_angle = np.random.uniform(0, 2 * np.pi)
                vx = vel_mag * np.cos(vel_angle)
                vy = vel_mag * np.sin(vel_angle)
                p.resetBaseVelocity(oid, [vx, vy, 0], physicsClientId=self.CLIENT)


    def _updateAndStoreKinematicInformation(self):
        #Agent#
        self.agent_pos, _ = p.getBasePositionAndOrientation(self.AGENT_ID, physicsClientId=self.CLIENT)
        self.agent_vel, _ = map(np.array, p.getBaseVelocity(self.AGENT_ID, physicsClientId=self.CLIENT))

        #Target#
        self.target_pos, _ = p.getBasePositionAndOrientation(self.TARGET_ID, physicsClientId=self.CLIENT)
        self.target_vel, _ = map(np.array, p.getBaseVelocity(self.TARGET_ID, physicsClientId=self.CLIENT))


    def _getAgentStateVector(self):
        state = np.hstack([self.agent_pos,
                           self.agent_vel])
        return state.reshape(6)


    def _getTargetStateVector(self):
        state = np.hstack([self.target_pos,
                           self.target_vel])
        return state.reshape(6)


    def _actionSpace(self):
        size = 2
        act_lower_bound = np.array(-1*np.ones(size))
        act_upper_bound = np.array(+1*np.ones(size))
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)


    def _observationSpace(self):
        obs_lower_bound = np.array([0, 0, -1, 0,  # agent: x, y, angle, vel
                                    0, 0,  # target: x, y
                                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # 9 ray distances
                                    -1, 0, -1])  # beta, distance, alpha
        
        obs_higher_bound = np.array([1, 1, 1, 1,
                                     1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1])
        
        return spaces.Box(low=obs_lower_bound, high=obs_higher_bound, dtype=np.float32)


    def _computeObs(self):
        obs_20 = np.zeros(18)

        obs_agent = self._getAgentStateVector()
        obs_target = self._getTargetStateVector()

        agent_vel = np.linalg.norm(obs_agent[3:5])
        agent_angle = np.arctan2(obs_agent[4], obs_agent[3])

        obs_20[0] = obs_agent[0] / self.w
        obs_20[1] = obs_agent[1] / self.h
        obs_20[2] = agent_angle / np.pi
        obs_20[3] = agent_vel / self.agent_vel_max
        obs_20[4] = obs_target[0] / self.w
        obs_20[5] = obs_target[1] / self.h

        ray_dist_norm = self._computeRayDistances(obs_agent[0], obs_agent[1], agent_angle)
        obs_20[6:15] = ray_dist_norm

        beta = np.arctan2(obs_target[1]-obs_agent[1], obs_target[0]-obs_agent[0])
        obs_20[15] = beta / np.pi

        agent_target_diff = [obs_target[0]-obs_agent[0], obs_target[1]-obs_agent[1]]
        agent_target_diff = np.linalg.norm(agent_target_diff)
        obs_20[16] = agent_target_diff / np.linalg.norm([self.w, self.h])

        a = [np.cos(agent_angle), np.sin(agent_angle)]
        b = [(obs_target[0] - obs_agent[0]), (obs_target[1] - obs_agent[1])]
        dot = np.dot(a, b)
        cross = a[0] * b[1] - a[1] * b[0]
        alpha = np.arccos(np.clip(dot / (agent_target_diff + 1e-8), -1.0, 1.0))
        if cross < 0:
            alpha = -alpha
        obs_20[17] = alpha / np.pi

        return obs_20


    def _computeRayDistances(self,
                             x,
                             y,
                             vel_angle
                             ):
        ray_from = []
        ray_to = []
        start_angle = vel_angle - np.pi/3
        eps = self.agent_radius + 1e-3

        for r in range(self.num_rays):
            angle = start_angle + r * (np.pi / 12)
            dx = np.cos(angle)
            dy = np.sin(angle)

            from_pos = [x + eps * dx, y + eps * dy, self.z]
            to_pos = [x + (self.r_cap + eps) * dx, y + (self.r_cap + eps) * dy, self.z]

            ray_from.append(from_pos)
            ray_to.append(to_pos)

        ray_results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
        ray_dist_norm = np.zeros(self.num_rays)

        for i, r in enumerate(ray_results):
            ray_dist_norm[i] = r[2]

        return ray_dist_norm


    def _preprocessAction(self,
                          action
                          ):
        lambda_v = action[0]
        lambda_theta = action[1]

        state = self._getAgentStateVector()

        vx = state[3]
        vy = state[4]
        v_mag = np.linalg.norm([vx, vy])
        v_angle = np.arctan2(vy, vx)

        v_mag_1 = v_mag + lambda_v * self.del_vel_max
        v_mag_1 = np.clip(v_mag_1, 0, self.agent_vel_max)
        v_angle_1 = v_angle + lambda_theta * self.del_angle_max
        vx_1 = v_mag_1 * np.cos(v_angle_1)
        vy_1 = v_mag_1 * np.sin(v_angle_1)

        return np.array([vx_1, vy_1])

    def adaptive_filter(
            self,
            v_desired: np.ndarray,
            verbose: bool = False,
    ):
        """
        Adaptive filter that switches between CLF-only and CBF-only modes
        based on obstacle proximity.
        
        Modes:
        - 'switching': Switch between CLF and CBF based on distance
        - 'unified': Use both CLF and CBF simultaneously (relaxed CLF)
        - 'clf_only': Only CLF (original behavior)
        - 'cbf_only': Only CBF
        """
        # Get current state
        p_agent = np.array(self.agent_pos[:2])
        p_target = np.array(self.target_pos[:2])
        v_current = np.array(self.agent_vel[:2])
        
        # Get ray distances for obstacle detection
        agent_angle = np.arctan2(v_current[1], v_current[0])
        ray_dist_norm = self._computeRayDistances(p_agent[0], p_agent[1], agent_angle)
        ray_distances = ray_dist_norm * self.r_cap
        min_ray_distance = np.min(ray_distances)
        
        # Compute CLF
        p_diff = p_agent - p_target
        V_current = 0.5 * np.dot(p_diff, p_diff)
        dV_dt_grad = p_diff
        
        # Compute CBF
        h_current = 0.5 * (min_ray_distance - self.safety_distance)**2
        min_ray_idx = np.argmin(ray_distances)
        start_angle = agent_angle - np.pi/3
        ray_angle = start_angle + min_ray_idx * (np.pi / 12)
        obstacle_direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
        dh_dt_grad = -(min_ray_distance - self.safety_distance) * obstacle_direction
        
        # Decide which mode to use
        if self.filter_mode == 'switching':
            # Switch between CLF-only and CBF-only based on distance
            if min_ray_distance < self.cbf_activation_distance:
                active_mode = 'CBF'
                v_final, info = self._cbf_only_filter(v_desired, h_current, dh_dt_grad, verbose)
            else:
                active_mode = 'CLF'
                v_final, info = self._clf_only_filter(v_desired, V_current, dV_dt_grad, verbose)
                
        elif self.filter_mode == 'unified':
            # Use both CLF and CBF with relaxation
            active_mode = 'UNIFIED'
            v_final, info = self._unified_filter(v_desired, V_current, dV_dt_grad, 
                                                 h_current, dh_dt_grad, 
                                                 min_ray_distance, verbose)
            
        elif self.filter_mode == 'clf_only':
            active_mode = 'CLF'
            v_final, info = self._clf_only_filter(v_desired, V_current, dV_dt_grad, verbose)
            
        elif self.filter_mode == 'cbf_only':
            active_mode = 'CBF'
            v_final, info = self._cbf_only_filter(v_desired, h_current, dh_dt_grad, verbose)
        
        else:
            # Default to no filter
            active_mode = 'NONE'
            v_final = v_desired
            info = {
                'qp_success': True,
                'V_current': V_current,
                'h_current': h_current,
                'min_ray_distance': min_ray_distance,
                'v_desired': v_desired.copy(),
                'v_safe': v_final.copy(),
                'action_deviation': 0.0,
            }
        
        # Add mode information
        info['active_mode'] = active_mode
        info['min_ray_distance'] = min_ray_distance
        
        return v_final, info
    
    def _clf_only_filter(self, v_desired, V_current, dV_dt_grad, verbose=False):
        """CLF-only QP filter for target tracking"""
        v_safe = cp.Variable(2)
        
        # CLF constraint
        clf_constraint = dV_dt_grad @ v_safe <= -self.gamma_clf * V_current
        
        # Velocity limits
        constraints = [
            v_safe >= -self.agent_vel_max,
            v_safe <= self.agent_vel_max,
            clf_constraint
        ]
        
        # Objective: minimize deviation from desired velocity
        objective = cp.Minimize(0.5 * self.lambda_clf * cp.sum_squares(v_safe - v_desired))
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                v_final = v_safe.value
                success = True
            else:
                v_final = v_desired
                success = False
                if verbose:
                    print(f"CLF QP failed with status: {problem.status}")
        except Exception as e:
            v_final = v_desired
            success = False
            if verbose:
                print(f"CLF QP solver exception: {e}")
        
        dV_dt_final = dV_dt_grad @ v_final
        
        info = {
            'qp_success': success,
            'V_current': V_current,
            'h_current': 0.0,
            'dV_dt': dV_dt_final,
            'dh_dt': 0.0,
            'clf_satisfied': dV_dt_final <= -self.gamma_clf * V_current + self.epsilon,
            'cbf_satisfied': True,  # Not applicable
            'v_desired': v_desired.copy(),
            'v_safe': v_final.copy(),
            'action_deviation': np.linalg.norm(v_final - v_desired),
        }
        
        return v_final, info
    
    def _cbf_only_filter(self, v_desired, h_current, dh_dt_grad, verbose=False):
        """CBF-only QP filter for obstacle avoidance"""
        v_safe = cp.Variable(2)
        
        # CBF constraint
        cbf_constraint = dh_dt_grad @ v_safe >= -self.alpha_cbf * h_current
        
        # Velocity limits
        constraints = [
            v_safe >= -self.agent_vel_max,
            v_safe <= self.agent_vel_max,
        ]
        
        # Only add CBF constraint if h >= 0 (inside safe set)
        if h_current >= 0:
            constraints.append(cbf_constraint)
        
        # Objective: minimize deviation from desired velocity
        objective = cp.Minimize(0.5 * self.lambda_clf * cp.sum_squares(v_safe - v_desired))
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                v_final = v_safe.value
                success = True
            else:
                v_final = v_desired
                success = False
                if verbose:
                    print(f"CBF QP failed with status: {problem.status}")
        except Exception as e:
            v_final = v_desired
            success = False
            if verbose:
                print(f"CBF QP solver exception: {e}")
        
        dh_dt_final = dh_dt_grad @ v_final
        
        info = {
            'qp_success': success,
            'V_current': 0.0,
            'h_current': h_current,
            'dV_dt': 0.0,
            'dh_dt': dh_dt_final,
            'clf_satisfied': True,  # Not applicable
            'cbf_satisfied': (h_current < 0) or (dh_dt_final >= -self.alpha_cbf * h_current - self.epsilon),
            'v_desired': v_desired.copy(),
            'v_safe': v_final.copy(),
            'action_deviation': np.linalg.norm(v_final - v_desired),
        }
        
        return v_final, info
    
    def _unified_filter(self, v_desired, V_current, dV_dt_grad, 
                       h_current, dh_dt_grad, min_ray_distance, verbose=False):
        """Unified CLF-CBF QP with relaxation (original implementation)"""
        v_safe = cp.Variable(2)
        delta = cp.Variable()
        
        # CLF constraint (soft)
        clf_constraint = dV_dt_grad @ v_safe <= -self.gamma_clf * V_current + delta
        
        # CBF constraint (hard, only when obstacles are close)
        cbf_active = min_ray_distance < self.cbf_activation_distance
        
        constraints = [
            v_safe >= -self.agent_vel_max,
            v_safe <= self.agent_vel_max,
            clf_constraint
        ]
        
        if cbf_active and h_current >= 0:
            cbf_constraint = dh_dt_grad @ v_safe >= -self.alpha_cbf * h_current
            constraints.append(cbf_constraint)
        
        # Objective
        objective = cp.Minimize(
            0.5 * self.lambda_clf * cp.sum_squares(v_safe - v_desired) + 
            1000.0 * cp.square(delta)
        )
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                v_final = v_safe.value
                delta_val = delta.value
                success = True
            else:
                v_final = v_desired
                delta_val = 0.0
                success = False
                if verbose:
                    print(f"Unified QP failed with status: {problem.status}")
        except Exception as e:
            v_final = v_desired
            delta_val = 0.0
            success = False
            if verbose:
                print(f"Unified QP solver exception: {e}")
        
        dV_dt_final = dV_dt_grad @ v_final
        dh_dt_final = dh_dt_grad @ v_final if cbf_active else 0.0
        
        info = {
            'qp_success': success,
            'V_current': V_current,
            'h_current': h_current,
            'dV_dt': dV_dt_final,
            'dh_dt': dh_dt_final,
            'delta': delta_val,
            'clf_satisfied': dV_dt_final <= -self.gamma_clf * V_current + delta_val + self.epsilon,
            'cbf_satisfied': (not cbf_active) or (dh_dt_final >= -self.alpha_cbf * h_current - self.epsilon),
            'v_desired': v_desired.copy(),
            'v_safe': v_final.copy(),
            'action_deviation': np.linalg.norm(v_final - v_desired),
        }
        
        return v_final, info

    def _computeReward(self):
        obs = self._computeObs()
        dist = self.AGENT_TARGET_DIST
        vel_norm = np.linalg.norm(self.agent_vel[:2] / self.agent_vel_max)
        ray_dist = obs[6:15]

        r_d = self.k_D * (self.LAST_AGENT_TARGET_DIST - self.AGENT_TARGET_DIST)
        r_theta = self.k_theta * (np.abs(self.LAST_ALPHA) - np.abs(obs[17])) * np.pi
        r_obs = self.k_obs * np.sum(obs[6:15] - self.prev_ray_dist)
        r_v = self.k_v * (obs[3] * self.agent_vel_max - self.v_c)

        r = r_d + r_theta + r_obs + r_v

        if np.any(ray_dist < 0.5):
            r -= self.collision_penalty
        elif dist < 5:
            r += self.success_penalty

        return r


    def _computeTerminated(self):
        dist_to_target = np.linalg.norm(np.array(self.agent_pos[0:2]) - np.array(self.target_pos[0:2]))
        return dist_to_target < 0.5

    def _computeTruncated(self):
        angle = np.arctan2(self.agent_vel[1], self.agent_vel[0])
        ray_dist = self._computeRayDistances(self.agent_pos[0], self.agent_pos[1], angle)

        collision = np.any(ray_dist < 0.05)
        if collision:
            self.termination_cause = "collision"

        out_of_bound = (self.agent_pos[0] > self.w) or (self.agent_pos[1] > self.h) or (self.agent_pos[0] < 0) or (
                    self.agent_pos[1] < 0)
        if out_of_bound:
            self.termination_cause = "out of bound"

        timeout = (self.step_counter > self.EPISODE_LEN_SEC)
        if timeout:
            self.termination_cause = "timeout"

        return collision or timeout or out_of_bound

    def _computeInfo(self, terminated=False, truncated=False):
        success = terminated and not truncated
        info = {
            "success": success,
            "termination": self.termination_cause,
        }
        if self.filter_info is not None:
            info.update({
                "active_mode": self.filter_info.get('active_mode', 'NONE'),
                "cbf_mode_rate": self.cbf_mode_count / max(self.qp_call_count, 1),
                "clf_mode_rate": self.clf_mode_count / max(self.qp_call_count, 1),
                "qp_success_rate": self.qp_success_count / max(self.qp_call_count, 1),
                "avg_action_deviation": self.total_action_deviation / max(self.qp_call_count, 1),
                "last_V": self.filter_info.get('V_current', 0),
                "last_h": self.filter_info.get('h_current', 0),
                "last_min_ray_dist": self.filter_info.get('min_ray_distance', 0),
            })
        self.termination_cause = None
        return info

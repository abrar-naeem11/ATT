import os
from sys import platform
import time
import collections
from collections import deque
from datetime import datetime
import xml.etree.ElementTree as etxml
from PIL import Image
# import pkgutil
# egl = pkgutil.get_loader('eglRenderer')
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from DSLPIDControl import DSLPIDControl

class AOATTAviary(gym.Env):
    """Autonomous Obstacle Avoidance and Target Tracking of UAV Gym Environment"""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_agent_xy=None,
                 initial_target_xy=None,
                 initial_agent_yaw=None,
                 initial_target_yaw=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 output_folder='results',
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.VEL
                 ):
        self.EPISODE_LEN_SEC = 100

        #
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        #

        #Observation Parameters
        self.w = 300
        self.h = 300
        self.agent_vel_max = 1
        self.target_vel_max = np.sqrt(2)
        self.num_rays = 9
        self.r_cap = 15

        #Action Parameters
        self.del_vel_max = 0.5
        self.del_angle_max = np.pi / 6

        #Reward Parameters
        self.k_T = 10
        self.k_D = 0.05
        self.k_theta = 0.2
        self.k_obs = 1.5
        self.r_clip = 0.5
        self.d_thr = 10
        self.k_V = 0.2
        self.v_c = 0.5

        #Create integrated controllers#
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        self.agent_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.target_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # Constants#
        self.G = 9.8
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in AOTTAviary.__init__(), pyb_freq is not a multiple of ctrl_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ

        #Options#
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.PHYSICS = physics
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder

        #### Load the drone properties from the .urdf file #########
        self.M, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3 = self._parseURDFParameters()
        print(
            "[INFO] AOATTAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
                self.M, self.L, self.J[0, 0], self.J[1, 1], self.J[2, 2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO,
                self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2],
                self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))

        #### Compute constants #####################################
        self.GRAVITY = self.G * self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM ** 2)
        self.MAX_XY_TORQUE = (2 * self.L * self.KF * self.MAX_RPM ** 2) / np.sqrt(2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM ** 2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt(
            (15 * self.MAX_RPM ** 2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

        #Conncet to PyBullet#
        if self.GUI:
            self.CLIENT = p.connect(p.GUI)
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                      p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_SHADOWS]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=150,
                                         cameraYaw=0,
                                         cameraPitch=-89.99,
                                         cameraTargetPosition=[150., 150., 1.],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
        else:
            self.CLIENT = p.connect(p.DIRECT)

        #Set Initial Position#
        self.agent_XY_init = initial_agent_xy
        self.agent_YAW_init = initial_agent_yaw
        #Randomize agent position#
        self.INIT_AGENT_XY = np.random.uniform(low=[10.0, 10.0], high=[290.0, 290.0])
        self.INIT_AGENT_YAW = np.random.uniform(-np.pi, np.pi)

        self.target_XY_init = initial_target_xy
        self.target_YAW_init = initial_target_yaw
        #Randomize target position#
        self.INIT_TARGET_XY = np.random.uniform(low=[10.0, 10.0], high=[200.0, 200.0])
        self.INIT_TARGET_YAW = np.random.uniform(-np.pi, np.pi)

        self.z = 1.0
        self.roll = 0.0
        self.pitch = 0.0

        #Create action and observation spaces#
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        #Housekeeping#
        self._housekeeping()

        #Update and Store the drones kinematic information#
        self._updateAndStoreKinematicInformation()


    #####

    def reset(self,
              seed: int = None,
              options: dict = None,
              ):
        p.resetSimulation(physicsClientId=self.CLIENT)
        p.removeAllUserDebugItems()

        #Agent#
        if self.agent_XY_init is not None:
            self.INIT_AGENT_XY = self.agent_XY_init

        if self.agent_YAW_init is not None:
            self.INIT_AGENT_YAW = self.agent_YAW_init

        #Target#
        if self.target_XY_init is not None:
            self.INIT_TARGET_XY = self.target_XY_init

        if self.target_YAW_init is not None:
            self.INIT_TARGET_YAW = self.target_YAW_init

        #HouseKeeping#
        self._housekeeping()

        #Update and Store the drones kinematic information#
        self._updateAndStoreKinematicInformation()

        #Return the initial observation#
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info


    #####


    def step(self,
             action
             ):
        #Save, preprocess, and clip the action to max. RPM#
        clipped_action = np.reshape(self._preprocessAction(action), 4)

        #Repeat for as many as the aggregate physics steps#
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self._physics(clipped_action)

            #PyBullet computes the new state#
            p.stepSimulation(physicsClientId=self.CLIENT)

        #Store Last Distance between agent and target#
        self.LAST_AGENT_TARGET_DIST = np.linalg.norm([self.agent_pos[0]-self.target_pos[0], self.agent_pos[1]-self.target_pos[1]])

        #Update and store the drones kinematic information#
        self._updateAndStoreKinematicInformation()

        #Store Current Distance between agent and target#
        self.AGENT_TARGET_DIST = np.linalg.norm([self.agent_pos[0]-self.target_pos[0], self.agent_pos[1]-self.target_pos[1]])

        #Prepare the return values#
        obs = self._computeObs()
        reward = self._computeReward()
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()

        #Advance the step counter#
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return obs, reward, terminated, truncated, info


    #####


    def render(self):
        if self.first_render_call and not self.GUI:
            print(
                "[WARNING] AOATTAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface."
            )
            self.first_render_call = False
        print("\n[INFO] AOATTAviary.render() --- it {:04d}".format(self.step_counter),
              "--- wall-clock time {:.1f}s,".format(time.time() - self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f})".format(self.step_counter * self.PYB_TIMESTEP, self.PYB_FREQ,
                                                                   (self.step_counter * self.PYB_TIMESTEP) / (
                                                                       time.time() - self.RESET_TIME))
                                                                   )
        print("[INFO] AOATTAviary.render() --- agent",
              "--- x{:+06.2f}, y{:+06.2f}, z {:+06.2f}".format(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]),
              "--- velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.agent_vel[0], self.agent_vel[1], self.agent_vel[2]),
              "--- roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.agent_rpy[0] * self.RAD2DEG,
                                                                         self.agent_rpy[1] * self.RAD2DEG,
                                                                         self.agent_rpy[2] * self.RAD2DEG),
              "--- angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ---".format(self.agent_ang_v[0], self.agent_ang_v[1], self.agent_ang_v[2]))

        print("[INFO] AOATTAviary.render() --- target",
              "--- x{:+06.2f}, y{:+06.2f}, z {:+06.2f}".format(self.target_pos[0], self.target_pos[1], self.target_pos[2]),
              "--- velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.target_vel[0], self.target_vel[1], self.target_vel[2]),
              "--- roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.target_rpy[0] * self.RAD2DEG,
                                                                         self.target_rpy[1] * self.RAD2DEG,
                                                                         self.target_rpy[2] * self.RAD2DEG),
              "--- angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ---".format(self.target_ang_v[0], self.target_ang_v[1], self.target_ang_v[2]))


    #####


    def close(self):
        p.disconnect(physicsClientId=self.CLIENT)


    #####


    def getPyBulletClient(self):
        return self.CLIENT


    #####


    def getAgentId(self):
        return self.AGENT_ID


    #####


    def getTargetId(self):
        return self.TARGET_ID


    #####



    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        self.first_render_call = True

        #### Initialize the drones kinemaatic information ##########
        #Agent#
        self.agent_pos = np.zeros(3)
        self.agent_quat = np.zeros(4)
        self.agent_rpy = np.zeros(3)
        self.agent_vel = np.zeros(3)
        self.agent_ang_v = np.zeros(3)

        #Target#
        self.target_pos = np.zeros(3)
        self.target_quat = np.zeros(4)
        self.target_rpy = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.target_ang_v = np.zeros(3)

        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)

        self.AGENT_ID = p.loadURDF(os.path.join("assets", self.URDF),
                                   basePosition=[self.INIT_AGENT_XY[0], self.INIT_AGENT_XY[1], self.z],
                                   baseOrientation=p.getQuaternionFromEuler([self.roll, self.pitch, self.INIT_AGENT_YAW]),
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=self.CLIENT
                                   )

        self.TARGET_ID = p.loadURDF(os.path.join("assets", self.URDF),
                                    basePosition=[self.INIT_TARGET_XY[0], self.INIT_TARGET_XY[1], self.z],
                                    baseOrientation=p.getQuaternionFromEuler([self.roll, self.pitch, self.INIT_TARGET_YAW]),
                                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                                    physicsClientId=self.CLIENT
                                    )

        # --- Visual-only markers (no physics) ---
        blue_vis = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[0, 0, 1, 1], physicsClientId=self.CLIENT)
        red_vis = p.createVisualShape(p.GEOM_SPHERE, radius=1, rgbaColor=[1, 0, 0, 1], physicsClientId=self.CLIENT)
        
        self.AGENT_MARKER = p.createMultiBody(
            baseMass=0,  # static
            baseCollisionShapeIndex=-1,  # NO collision => visual only
            baseVisualShapeIndex=blue_vis,
            basePosition=[self.INIT_AGENT_XY[0], self.INIT_AGENT_XY[1], self.z + 0.05],
            physicsClientId=self.CLIENT
        )

        self.TARGET_MARKER = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=red_vis,
            basePosition=[self.INIT_TARGET_XY[0], self.INIT_TARGET_XY[1], self.z + 0.05],
            physicsClientId=self.CLIENT
        )

        self._addObstacles()


    #####


    def _updateAndStoreKinematicInformation(self):
        #Agent#
        self.agent_pos, self.agent_quat = p.getBasePositionAndOrientation(self.AGENT_ID, physicsClientId=self.CLIENT)
        self.agent_rpy = p.getEulerFromQuaternion(self.agent_quat)
        self.agent_vel, self.agent_ang_v = map(np.array, p.getBaseVelocity(self.AGENT_ID, physicsClientId=self.CLIENT))

        #Target#
        self.target_pos, self.target_quat = p.getBasePositionAndOrientation(self.TARGET_ID, physicsClientId=self.CLIENT)
        self.target_rpy = p.getEulerFromQuaternion(self.target_quat)
        self.target_vel, self.target_ang_v = map(np.array, p.getBaseVelocity(self.TARGET_ID, physicsClientId=self.CLIENT))

        # keep markers on top of the drones
        p.resetBasePositionAndOrientation(
            self.AGENT_MARKER,
            [self.agent_pos[0], self.agent_pos[1], self.z + 0.2],
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        p.resetBasePositionAndOrientation(
            self.TARGET_MARKER,
            [self.target_pos[0], self.target_pos[1], self.z + 0.2],
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        #self._updateCamera()


    #####


    def _getAgentStateVector(self):
        state = np.hstack([self.agent_pos, self.agent_quat, self.agent_rpy,
                           self.agent_vel, self.agent_ang_v])
        return state.reshape(16)


    #####


    def _getTargetStateVector(self):
        state = np.hstack([self.target_pos, self.target_quat, self.target_rpy,
                           self.target_vel, self.target_ang_v])
        return state.reshape(16)


    #####


    def _physics(self,
                 rpm
                 ):
        forces = np.array(rpm ** 2) * self.KF
        torques = np.array(rpm ** 2) * self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.AGENT_ID,
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.AGENT_ID,
                             -1,
                             torqueObj=[0, 0, z_torque],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )


    #####


    def _addObstacles(self):

        #Radius 1#
        p.loadURDF("sphere2.urdf", [50, 75, 1])
        p.loadURDF("sphere2.urdf", [50, 150, 1])
        p.loadURDF("sphere2.urdf", [75, 125, 1])
        p.loadURDF("sphere2.urdf", [75, 175, 1])
        p.loadURDF("sphere2.urdf", [150, 35, 1])
        p.loadURDF("sphere2.urdf", [150, 50, 1])
        p.loadURDF("sphere2.urdf", [150, 100, 1])
        p.loadURDF("sphere2.urdf", [175, 75, 1])
        p.loadURDF("sphere2.urdf", [175, 125, 1])
        p.loadURDF("sphere2.urdf", [175, 150, 1])

        #Radius 2#
        p.loadURDF("sphere2.1.urdf", [25, 50, 1])
        p.loadURDF("sphere2.1.urdf", [25, 125, 1])
        p.loadURDF("sphere2.1.urdf", [75, 50, 1])
        p.loadURDF("sphere2.1.urdf", [125, 85, 1])
        p.loadURDF("sphere2.1.urdf", [125, 150, 1])

        #Radius 3#
        p.loadURDF("sphere2.2.urdf", [125, 150, 1])


    #####


    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(os.path.join("assets", self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3


    #####

    def _actionSpace(self):
        #Returns the action space of the environment

        size = 2 # Action Type is fraction of velocity and action change
        act_lower_bound = np.array(-1*np.ones(size))
        act_upper_bound = np.array(+1*np.ones(size))

        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    #####

    def _observationSpace(self):
        #Returns the observation space of the environment

        #Agent
        #Position
        agent_x_low = 0
        agent_x_high = 1

        agent_y_low = 0
        agent_y_high = 1

        #Angle
        agent_yaw_low = -1
        agent_yaw_high = 1

        #Velocity Magnitude
        agent_vel_low = 0
        agent_vel_high = 1

        #Target
        #Position
        target_x_low = 0
        target_x_high = 1

        target_y_low = 0
        target_y_high = 1

        #Angle
        target_yaw_low = -1
        target_yaw_high = 1

        #Velocity Magnitude
        target_vel_low = 0
        target_vel_high = 1

        #Ray Distance
        ray_1_dis_low = 0
        ray_1_dis_high = 1

        ray_2_dis_low = 0
        ray_2_dis_high = 1

        ray_3_dis_low = 0
        ray_3_dis_high = 1

        ray_4_dis_low = 0
        ray_4_dis_high = 1

        ray_5_dis_low = 0
        ray_5_dis_high = 1

        ray_6_dis_low = 0
        ray_6_dis_high = 1

        ray_7_dis_low = 0
        ray_7_dis_high = 1

        ray_8_dis_low = 0
        ray_8_dis_high = 1

        ray_9_dis_low = 0
        ray_9_dis_high = 1

        #Relative angle between the agent-target connection direction and the horizontal direction
        beta_low = -1
        beta_high = 1

        #Relative distance between the agent-target
        dis_low = 0
        dis_high = 1

        #Angle difference between the agent-target connection line and the agent yaw
        alpha_low = -1
        alpha_high = 1

        obs_lower_bound = np.array([agent_x_low, agent_y_low, agent_yaw_low, agent_vel_low,
                                    target_x_low, target_y_low, target_yaw_low, target_vel_low,
                                    ray_1_dis_low, ray_2_dis_low, ray_3_dis_low, ray_4_dis_low,
                                    ray_5_dis_low, ray_6_dis_low, ray_7_dis_low, ray_8_dis_low,
                                    ray_9_dis_low, beta_low, dis_low, alpha_low
                                    ])

        obs_higher_bound = np.array([agent_x_high, agent_y_high, agent_yaw_high, agent_vel_high,
                                     target_x_high, target_y_high, target_yaw_high, target_vel_high,
                                     ray_1_dis_high, ray_2_dis_high, ray_3_dis_high, ray_4_dis_high,
                                     ray_5_dis_high, ray_6_dis_high, ray_7_dis_high, ray_8_dis_high,
                                     ray_9_dis_high, beta_high, dis_high, alpha_high
                                     ])

        return spaces.Box(low=obs_lower_bound, high=obs_higher_bound, dtype=np.float32)


    #####


    def _computeObs(self):
        #Observation Space of size 20
        obs_20 = np.zeros(20)

        obs_agent = self._getAgentStateVector()
        obs_target = self._getTargetStateVector()

        agent_vel = np.linalg.norm(obs_agent[10:12])
        agent_angle = np.arctan2(obs_agent[11], obs_agent[10])

        target_vel = np.linalg.norm(obs_target[10:12])
        target_angle = np.arctan2(obs_target[11], obs_target[10])

        obs_20[0] = obs_agent[0] / self.w
        obs_20[1] = obs_agent[1] / self.h
        obs_20[2] = agent_angle / np.pi
        obs_20[3] = agent_vel / self.agent_vel_max
        obs_20[4] = obs_target[0] / self.w
        obs_20[5] = obs_target[1] / self.h
        obs_20[6] = target_angle / np.pi
        obs_20[7] = target_vel / self.target_vel_max

        ray_dist_norm = self._computeRayDistances(obs_agent[0], obs_agent[1], agent_angle)
        obs_20[8:17] = ray_dist_norm

        beta = np.arctan2(obs_target[1]-obs_agent[1], obs_target[0]-obs_agent[0])
        obs_20[17] = beta / np.pi

        agent_target_diff = [obs_target[0]-obs_agent[0], obs_target[1]-obs_agent[1]]
        agent_target_diff = np.linalg.norm(agent_target_diff)
        obs_20[18] = agent_target_diff / np.linalg.norm([self.w, self.h])

        a = [np.cos(agent_angle), np.sin(agent_angle)]
        b = [(obs_target[0] - obs_agent[0]), (obs_target[1] - obs_agent[1])]
        dot = np.dot(a, b)
        cross = a[0] * b[1] - a[1] * b[0]  # determinant in 2D = a × b
        alpha = np.arccos(np.clip(dot / (agent_target_diff + 1e-8), -1.0, 1.0))
        if cross < 0:
            alpha = -alpha
        obs_20[19] = alpha / np.pi

        return obs_20


    #####


    def _preprocessAction(self,
                          action
                          ):
        lambda_v = action[0]
        lambda_theta = action[1]

        rpm = np.zeros(4)
        state = self._getAgentStateVector()

        #Current
        x = state[0]
        y = state[1]
        vx = state[10]
        vy = state[11]
        v_mag = np.linalg.norm([vx, vy])
        v_angle = np.arctan2(vy, vx)

        #Next
        v_mag_1 = v_mag + lambda_v * self.del_vel_max
        v_mag_1 = np.clip(v_mag_1, 0, self.agent_vel_max)
        v_angle_1 = v_angle + lambda_theta * self.del_angle_max
        vx_1 = v_mag_1 * np.cos(v_angle_1)
        vy_1 = v_mag_1 * np.sin(v_angle_1)
        x_1 = x + vx_1
        y_1 = y + vy_1

        rpm, _, _ = self.agent_ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                   cur_pos=state[0:3],
                                                   cur_quat=state[3:7],
                                                   cur_vel=state[10:13],
                                                   cur_ang_vel=state[13:16],
                                                   target_pos=np.array([x_1, y_1, self.z]),
                                                   target_rpy=np.array([0, 0, v_angle_1]),
                                                   target_vel=np.array([vx_1, vy_1, 0])
                                                   )

        return rpm


    #####


    def _computeReward(self):
        obs = self._computeObs()
        #Termination function#
        r_t = 0
        if np.linalg.norm(np.array(self.agent_pos[0:2]) - np.array(self.target_pos[0:2])) < 4*self.PROP_RADIUS:
            r_t = self.k_T
        elif np.any((obs[8:17]) < 0.05):
            r_t = -self.k_T

        #Distance function#
        r_d = self.k_D * (self.LAST_AGENT_TARGET_DIST - self.AGENT_TARGET_DIST)

        #Angle function#
        r_theta = (self.k_theta / np.pi) * ((np.pi / 2) - (obs[19] * np.pi))

        #Obstacle avoidance function#
        eps = 1e-8
        r = -self.k_obs * np.sum((1 / (obs[8:17] * self.r_cap + eps)) - (1 / self.r_cap))
        if r <= -self.r_clip:
            r_obs = -self.r_clip
        else:
            r_obs = r

        #Speed function#
        v_p = np.linalg.norm([self.agent_vel[0], self.agent_vel[1]])
        if v_p < self.v_c:
            r_v = -self.k_V * (self.v_c - v_p)
        else:
            r_v = 0

        #Straight line function#
        r_line = self.k_theta * (obs[12] * self.r_cap - self.d_thr) / self.d_thr

        #Total reward#
        r = r_t + r_d + r_theta + r_obs + r_v + r_line

        return r


    #####


    def _computeTerminated(self):
        dist_to_target = np.linalg.norm(np.array(self.agent_pos[0:2]) - np.array(self.target_pos[0:2]))
        return dist_to_target < 4*self.PROP_RADIUS


    #####


    def _computeTruncated(self):
        angle = np.arctan2(self.agent_vel[1], self.agent_vel[0])
        ray_dist = self._computeRayDistances(self.agent_pos[0], self.agent_pos[1], angle)

        collision = np.any(ray_dist < 0.05)

        out_of_bound = (self.agent_pos[0] > self.w) or (self.agent_pos[1] > self.h) or (self.agent_pos[0] < 0) or (self.agent_pos[1] < 0)

        timeout = ((self.step_counter / self.PYB_FREQ) > self.EPISODE_LEN_SEC)

        return collision or timeout or out_of_bound


    #####


    def _computeRayDistances(self,
                             x,
                             y,
                             vel_angle
                             ):
        ray_from = []
        ray_to = []
        start_angle = vel_angle - np.pi/3

        for r in range(self.num_rays):
            angle = start_angle + r * (np.pi / 12)
            dx = self.r_cap * np.cos(angle)
            dy = self.r_cap * np.sin(angle)

            eps_xy = self.COLLISION_R + 1e-3
            eps_z = 0.5 * self.COLLISION_H + 0.02

            from_pos = [x + eps_xy*np.cos(angle), y + eps_xy*np.sin(angle), self.z + eps_z]
            to_pos = [x + dx, y + dy, self.z + eps_z]

            ray_from.append(from_pos)
            ray_to.append(to_pos)

        ray_results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
        ray_dist_norm = np.zeros(self.num_rays)

        for i, r in enumerate(ray_results):
            ray_dist_norm[i] = r[2]

        return ray_dist_norm


    #####


    def _computeInfo(self):
        return {"answer": 42} #Unused


    #####

    def _updateCamera(self):
        x, y, z = self.agent_pos
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,  # zoom in/out
            cameraYaw=45,  # rotate around
            cameraPitch=-30,  # tilt
            cameraTargetPosition=[x, y, z],
            physicsClientId=self.CLIENT
        )
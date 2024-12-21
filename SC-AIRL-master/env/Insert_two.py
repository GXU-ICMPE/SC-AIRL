import gym
import pybullet as p
import pybullet_data
import math
import numpy as np
from gym import spaces
import os
import time
from math import sqrt
from gym.utils import seeding
from gail_airl_ppo.buffer import Buffer
import random

class ur5Env_1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_render=True, is_good_view=True, move_step=0.003):
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.move_step = move_step
        self.step_counter = 0
        self.stage = 0
        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._max_episode_steps = 1000

        self.reward = 0
        self.done = False
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.x_low_obs = 0.48 * 100
        self.x_high_obs = 0.68 * 100
        self.y_low_obs = -0.1 * 100
        self.y_high_obs = 0.1 * 100
        self.z_low_obs = 0.02 * 100
        self.z_high_obs = 0.125 * 100

        self.target_x_low_obs = 0.48 * 100
        self.target_x_high_obs = 0.72 * 100
        self.target_y_low_obs = -0.15 * 100
        self.target_y_high_obs = 0.15 * 100
        self.target_z_low_obs = 0.0 * 100
        self.target_z_high_obs = 0.17 * 100

        self.x1_low_obs = 0
        self.x1_high_obs = 1
        self.y1_low_obs = -1
        self.y1_high_obs = 1

        self.x2_low_obs = 0
        self.x2_high_obs = 1
        self.y2_low_obs = -1
        self.y2_high_obs = 1

        self.x3_low_obs = 0
        self.x3_high_obs = 1
        self.y3_low_obs = -1
        self.y3_high_obs = 1

        self.z = 0
        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1
        self.G = 0

        self.observation_space = spaces.Box(
            low=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_low_obs, self.target_y_low_obs, self.target_z_low_obs, -1.0
                 , self.z_low_action, self.y_low_action, self.x_low_action],
                dtype=float),
            high=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_high_obs, self.target_y_high_obs, self.target_z_high_obs, 3.0
                 ,self.z_high_action, self.y_high_action, self.x_high_action]),
            dtype=float
        )




        # if self.stage:
        #     self.action_space = spaces.Discrete(4)
        #     print(self.action_space.shape)
        # else:
        #     self.action_space = spaces.Box(low=np.array(
        #         [self.z_low_action, self.y_low_action, self.x_low_action]),
        #         high=np.array([self.z_high_action, self.y_high_action, self.x_high_action]),
        #         dtype=np.float32)

        self.action_space = spaces.Box(low=np.array(
            [self.z_low_action, self.y_low_action, self.x_low_action]),
            high=np.array([self.z_high_action, self.y_high_action, self.x_high_action]),
            dtype=np.float32)

        self.action_space1 = spaces.Box(low=np.array(
            [self.z_low_action, self.y_low_action, self.x_low_action]),
            high=np.array([self.z_high_action, self.y_high_action, self.x_high_action]),
            dtype=np.float32)

        self.change_space = spaces.Box(low=np.array(
            [0.0]),
            high=np.array([1.0]),
            dtype=np.float32)

        self.human = 0
        self.seed()


    def quaternion_rotation(self, q1, q2):
        r1 = q1[3]
        r2 = q2[3]
        v1 = np.array([q1[0], q1[1], q1[2]])
        v2 = np.array([q2[0], q2[1], q2[2]])

        r = r1 * r2 - np.dot(v1, v2)
        v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
        q = np.array([v[0], v[1], v[2], r])

        return q

    def get_position_r_vary(self, position, posture, E, x):
        a = [key for i in posture for key in i]
        need = np.array(p.getQuaternionFromEuler(E)).reshape(4, 1).tolist()
        b = [key for j in need for key in j]
        pos = np.array(self.quaternion_rotation(a, b)).reshape(4, 1)  # 此刻UR5末端的姿态

        matrix = np.array(p.getMatrixFromQuaternion(posture), dtype=float).reshape(3, 3)
        dy = x  # 沿着工具坐标系x轴往前x
        res = np.array(position, dtype=float).reshape(3, 1)
        res += matrix[:, 0].reshape(3, 1) * dy  # 此刻轴末端的位置

        matrix_new = np.array(p.getMatrixFromQuaternion(pos)).reshape(3, 3)
        dyy = -x
        position_need = np.array(res).reshape(3, 1)
        position_need += matrix_new[:, 0].reshape(3, 1) * dyy  # 此刻

        return position_need, pos, res

    def reset(self):
        self.action1 = 0
        self.action2 = 0
        self.action3 = 0
        self.step_counter = 0
        self.stage = 1
        self.done = False
        self.done1 = False
        self.G = 0
        self.finish_task_one = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

        # p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
        # p.loadURDF("tray/tray.urdf", basePosition=[0.62, 0, 0.0])
        # p.loadURDF("tray/traybox.urdf", [0.62, 0, 0.0])
        # self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.64, 0.0, 0.02])
        self.tray = p.loadURDF("D:\\pycharm_project\\xiangguangyu_SC_AIRL\\untitled\\pandaurdf\\tray_with_inserts.urdf", basePosition=[0.4, -0.2, 0.0],  useFixedBase=True)
        # self.cube2 = p.loadURDF("cube4.urdf", basePosition=[0.6, 0.07, 0.0], useFixedBase=True)
        # hole0 = [0.53, -0.01, z] hole1 = [0.65, -0.01, 0.05]
        self.cube1 = p.loadURDF("cube2.urdf", basePosition=[0.6, 0.08, 0.02])

        self.random_x = random.uniform(0.52,0.525)
        self.random_y = random.uniform(-0.05, -0.045)

        self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.6, -0.08, 0.02])

        # self.random_number = random.randint(1, 3)
        # if self.random_number == 1:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.62, -0.02, 0.02])
        # if self.random_number == 2:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.6, -0.02, 0.02])
        # if self.random_number == 3:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.64, -0.02, 0.02])


        # self.UR5 = p.loadURDF("D:\\pycharm_project\\untitled\\ur10urdf\\ur10.urdf", useFixedBase=True)
        self.UR5 = p.loadURDF("franka_panda/panda.urdf", basePosition=np.array([0, 0, 0]), useFixedBase=True)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.4,
            cameraYaw=75,
            cameraPitch=-30,
            cameraTargetPosition=[0.6, 0.0, 0.0]
        )
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs / 100, self.y_low_obs / 100, 0],
        #     lineToXYZ=[self.x_low_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs / 100, self.y_high_obs / 100, 0],
        #     lineToXYZ=[self.x_low_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs / 100, self.y_low_obs / 100, 0],
        #     lineToXYZ=[self.x_high_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs / 100, self.y_high_obs / 100, 0],
        #     lineToXYZ=[self.x_high_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100])
        #
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100],
        #     lineToXYZ=[self.x_high_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100],
        #     lineToXYZ=[self.x_high_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_low_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100],
        #     lineToXYZ=[self.x_low_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100])
        # p.addUserDebugLine(
        #     lineFromXYZ=[self.x_high_obs / 100, self.y_low_obs / 100, self.z_high_obs / 100],
        #     lineToXYZ=[self.x_high_obs / 100, self.y_high_obs / 100, self.z_high_obs / 100])

        joint_active_ids = [0, 1, 2, 3, 4, 5, 6, 9, 10]

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]
        for j in range(7):
            p.resetJointState(self.UR5, j, rest_poses[j])
        p.resetJointState(self.UR5, 9, 0.08)
        p.resetJointState(self.UR5, 10, 0.08)
        #
        init_position = [0.6, 0.0, 0.1]  # train position
        # init_position = [0.5308569, 0.08824733, 0.40342805]     # after train position
        init_postrue_vary = [0.0, 0.0, 0.0]

        init_posture = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)

        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.036)  # 初始姿态偏置
        jointposes = p.calculateInverseKinematics(self.UR5, 11, position_start, postrue_start, maxNumIterations=100)
        for j in range(7):
            p.resetJointState(self.UR5, j, jointposes[j])





        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)

        self.postrue = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        p.stepSimulation()
        self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
        self.observation_init = np.array(
            [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
             self.position_target_cube[0][0] * 100,self.position_target_cube[0][1] * 100, self.position_target_cube[0][2] * 100,
             self.stage, self.action1, self.action2, self.action3],
            dtype=float)
        self.cs0 = 2
        self.cs1 = 2
        self.cs2 = 2
        return self.observation_init
    #
    # def collect_ter_set(self, state):



    def step(self, action, change):

        self.cs0 = 2
        self.cs1 = 2
        self.cs2 = 2

        self.change = 0
        if change > 0.5:
            self.change = 1
        # if self.position_cube1[0][2] < 0.025:
        #     print(self.position_target_cube, self.position_cube1)
        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        # self.postrue = np.array(p.getLinkState(self.UR5, 6)[5], dtype=float).reshape(4, 1)
        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)

        # [0.65, -0.01, 0.05]
        if self.stage == 1 or self.stage == 0:
            self.distance = sqrt((self.position[0][0] - self.position_target_cube[0][0]) ** 2 + (
                        self.position[1][0] - self.position_target_cube[0][1]) ** 2 + (
                                             self.position[2][0] - self.position_target_cube[0][2]) ** 2)
            self.distance1 = sqrt(
                (self.position_target_cube[0][0] - 0.65) ** 2 + (self.position_target_cube[0][1] - (-0.01)) ** 2)

            self.distancez = sqrt((self.position_target_cube[0][2] - 0.02) ** 2)
        else:
            self.distance = sqrt((self.position[0][0] - self.position_cube1[0][0]) ** 2 + (
                        self.position[1][0] - self.position_cube1[0][1]) ** 2 + (
                                             self.position[2][0] - self.position_cube1[0][2]) ** 2)
            self.distance1 = sqrt(
                (self.position_cube1[0][0] - 0.53) ** 2 + (self.position_cube1[0][1] - (-0.01)) ** 2)

            self.distancez = sqrt((self.position_cube1[0][2] - 0.02) ** 2)

        # rayTos = np.array([self.position[0][0], self.position[1][0], self.position[2][0]-0.008])
        # results = p.rayTest(self.position, rayTos)
        # line = p.addUserDebugLine(self.position, rayTos, [0, 1, 0])

        self.action1 = action[0]
        self.action2 = action[1]
        self.action3 = action[2]

        dx = 0.005 * action[0]
        dy = 0.005 * action[1]
        dz = 0.005 * action[2]
        time.sleep(1/50)
        self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
        self.move(self.position_target, self.postrue)

        # p.removeUserDebugItem(line)

        if self.stage == 1:
            if self.change == 1:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                for i in range(10):
                    self.contact = p.getContactPoints(self.UR5, self.target_cube)

                if self.contact and self.distance < 0.01:
                    self.G = 1
                    self.stage = 0
                    self.cs0 = 1
                else:
                    self.G = 0
                    self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                    self.cs0 = 0
            else:
                self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
        elif self.stage == 0:
            if self.change == 1:
                self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                if self.distance1 <= 0.005:
                    if self.distancez <= 0.002:
                        self.G = 2
                        self.stage = 2
                        self.cs1 = 1
                else:
                    self.G = 0
                    self.stage = 1
                    self.cs1 = 0
            else:
                self.contact = p.getContactPoints(self.UR5, self.target_cube)
                if self.contact:
                    self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                else:
                    self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                    self.G = 0
                    self.stage = 1

        elif self.stage == 2:
            if self.change == 1:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                self.contact = p.getContactPoints(self.UR5, self.cube1)
                if self.contact and self.distance < 0.01:
                    self.G = 3
                    self.stage = 3
                    self.cs2 = 1
                else:
                    self.G = 2
                    self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                    self.cs2 = 0
            else:
                self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))

        elif self.stage == 3:
            self.contact = p.getContactPoints(self.UR5, self.cube1)
            if self.contact:
                self.move_gripper(self.position, self.postrue, list([0.01, 0.01]))
                if self.distance1 <= 0.005:
                    if self.distancez <= 0.002:
                        self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                        self.G = 4
            else:
                # if self.position_target_cube[0][2] < 0.04:
                self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
                self.G = 2
                self.stage = 2

        self.step_counter += 1


        state, reward, done, info = self._reward()

        return state, reward, done, info



    def straight(self, position, postrue, dy):  # 沿着轴方向前进，需要额外在使用动作时更新传入的position，postrue保持不变即可
        position_need = self.get_position_p(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def straight_y(self, position, postrue, dy):  # 沿着轴方向前进，需要额外在使用动作时更新传入的position，postrue保持不变即可
        position_need = self.get_position_y(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def straight_z(self, position, postrue, dy):  # 沿着轴方向前进，需要额外在使用动作时更新传入的position，postrue保持不变即可
        position_need = self.get_position_z(position, postrue, dy)
        self.move(position_need, postrue)

        return position_need, postrue

    def move(self, position, postrue):
        jointposes = p.calculateInverseKinematics(self.UR5, 11, position, postrue, maxNumIterations=100)
        p.setJointMotorControlArray(self.UR5, list([0,1,2,3,4,5,6,9,10]), p.POSITION_CONTROL
                                    , list(jointposes))
        n = 100
        while (n):
            p.stepSimulation()
            n = n - 1

    def move_gripper(self, position, postrue, move):

        p.setJointMotorControlArray(self.UR5, list([9,10]), p.POSITION_CONTROL, move)
        n = 100
        while (n):
            p.stepSimulation()
            n = n - 1

    def get_position_p(self, position, posture, dy):  # ur5的工具坐标系轴向为x,

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        # dy = 0.00005 沿着工具坐标系x轴向前进0.0005
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 0].reshape(3, 1) * dy
        return res

    def get_position_y(self, position, posture, dy):  # ur5的工具坐标系轴向为x,

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        # dy = 0.00005 沿着工具坐标系x轴向前进0.0005
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 1].reshape(3, 1) * dy
        return res

    def get_position_z(self, position, posture, dy):  # ur5的工具坐标系轴向为x,

        matrix = np.array(p.getMatrixFromQuaternion(posture)).reshape(3, 3)
        # dy = 0.00005 沿着工具坐标系x轴向前进0.0005
        res = np.array(position).reshape(3, 1)
        res += matrix[:, 2].reshape(3, 1) * dy
        return res

    def get_z(self):
        return self.z

    def _reward(self):
        # self.position_target_cube[0][0], self.position_target_cube[0][1]

        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)

        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        terminated = bool(
                self.position[0][0] * 100 < self.x_low_obs
                or self.position[0][0] * 100 > self.x_high_obs
                or self.position[1][0] * 100 < self.y_low_obs
                or self.position[1][0] * 100 > self.y_high_obs
                or self.position[2][0] * 100 < self.z_low_obs
                or self.position[2][0] * 100 > self.z_high_obs
            )

        info = 0
        self.reward = 0
        self.done1 = False
        # self.contact1 = p.getContactPoints(self.target_cube, self.cube1)
        if self.change == 1:
            if self.cs0 == 1:
                info = 1
                self.reward = 25
                self.done1 = True
                self.done = False
            else:
                info = -self.distance
                self.reward = -self.distance
                self.done1 = False
                self.done = False

            if self.cs1 == 1:
                info = 1
                self.reward = 25
                self.done1 = True
                self.done = False
            else:
                info = -self.distance1 - abs(self.distancez - 0.002)
                self.reward = -self.distance1 - abs(self.distancez - 0.002)
                self.done1 = False
                self.done = False

            if self.cs2 == 1:
                info = 1
                self.reward = 25
                self.done1 = True
                self.done = False
            else:
                info = -self.distance
                self.reward = -self.distance
                self.done1 = False
                self.done = False
        else:
            info = 0

        if terminated:
            self.reward = float(-10)
            self.done = True
            self.done1 = True
        elif self.step_counter >= self._max_episode_steps:
            self.reward = float(-10)
            self.done = True
            self.done1 = True
        elif self.stage == 1 and self.change == 0:
            self.reward = -self.distance
            self.done = False
        elif self.stage == 0 and self.change == 0:
            self.reward = -self.distance1 - abs(self.distancez - 0.002)
            self.done = False
        elif self.stage == 2 and self.change == 0:
            self.reward = -self.distance
            self.done = False
        elif self.stage == 3:
            self.reward = -self.distance1 - abs(self.distancez - 0.002)
            self.done = False
        elif self.G == 4:
            dx = 0.0
            dy = 0.0
            dz = 0.005
            for i in range(30):
                self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
                self.move(self.position_target, self.postrue)
            self.reward = 50
            self.done = True





        # hole0 = [0.53, -0.01, z] hole1 = [0.65, -0.01, 0.05]
        if self.stage == 1:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_target_cube[0][0] * 100, self.position_target_cube[0][1] * 100,
                 self.position_target_cube[0][2] * 100, self.stage, 0.0, 0.0, 0.0],
                dtype=float)
        elif self.stage == 0:
            self.observation = np.array(
                [self.position_target_cube[0][0] * 100, self.position_target_cube[0][1] * 100,
                 self.position_target_cube[0][2] * 100,
                 0.65 * 100, -0.01 * 100, 0.017 * 100, self.stage, 0.0, 0.0, 0.0],
                dtype=float)
        elif self.stage == 2:
            self.observation = np.array(
                [self.position[0][0] * 100, self.position[1][0] * 100, self.position[2][0] * 100,
                 self.position_cube1[0][0] * 100, self.position_cube1[0][1] * 100,
                 self.position_cube1[0][2] * 100, self.stage, 0.0, 0.0, 0.0],
                dtype=float)
        elif self.stage == 3:
            self.observation = np.array(
                [self.position_cube1[0][0] * 100, self.position_cube1[0][1] * 100,
                 self.position_cube1[0][2] * 100,
                 0.53 * 100, -0.01 * 100, 0.017 * 100, self.stage, 0.0, 0.0, 0.0],
                dtype=float)

        self.reward += info
        self.information()
        return self.observation, self.reward, self.done, info


    def infomation(self):
        return self.stage

    def information(self):
        return self.done1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        p.disconnect()

    def collect_demo(self, buffer, env, state, action):


        next_state, reward, done, _ = env.step(action)

        if self.stage == 0 and self.done1:
            buffer.append(state, action, reward, self.done1, next_state)
            self.done1 = 0
        else:
            buffer.append(state, action, reward, done, next_state)
        state = next_state


        return state, done

#
if __name__ == "__main__":
    env = ur5Env_1()
    state = env.reset()
    while True:
        env.step([0,0,0], 0)
        p.stepSimulation()
#
#     # from stable_baselines3.common.env_checker import check_env
#     #
#     # check_env(env)
#
#     # env = ur5Env_1(is_render=True)
#     #   #z y x
#     #
#     # for i in range(2000):
#     #     action = np.array([0, 0, 1])
#     #     env.step(action)
#     # # for i in range(10):
#     # #     env.step(action1)
#     step = 0
#     step1 = 0
#     done = 0
#     stage = 1
#     buffer_size = 6000
#     buffer_size1 = 4500
#     buffer = Buffer(
#         buffer_size=buffer_size,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space.shape,
#         device='cuda'
#     )
#
#
#     buffer1 = Buffer(
#         buffer_size=buffer_size1,
#         state_shape=env.observation_space.shape,
#         action_shape=env.action_space1.shape,
#         device='cuda'
#     )
#
#     # while True:
#     #
#     #     for i in range(55):
#     #         action = np.array([0, 0, 1])
#     #         state, done, stage = env.collect_demo(buffer, env, state, action)
#     #         step += 1
#     #     for i in range(80):
#     #         if not stage:
#     #             break
#     #         action = np.array([0, 1, 0])
#     #         state, done, stage = env.collect_demo(buffer, env, state, action)
#     #         step += 1
#     #     if step >= buffer_size:
#     #         print('Collect push done')
#     #         buffer.save(os.path.join(
#     #             'buffers',
#     #             'ur10-push',
#     #             f'ur10-v3-00915.pth'
#     #         ))
#     #
#     #     for i in range(43):
#     #         action = np.array([0, 1, 0])
#     #         state, done, stage = env.collect_demo(buffer1, env, state, action)
#     #         step1 += 1
#     #     for i in range(500):
#     #         if done:
#     #             break
#     #         action = np.array([1, 0, 0])
#     #         state, done, stage = env.collect_demo(buffer1, env, state, action)
#     #         step1 += 1
#     #     print(step1)
#     #     if step1 >= buffer_size1:
#     #         print('Collect reach done')
#     #         buffer1.save(os.path.join(
#     #             'buffers',
#     #             'ur10-reach',
#     #             f'ur10-v3-00915.pth'
#     #         ))
#     #     env.reset()
#     while True:
#         p.stepSimulation()
#         keys = p.getKeyboardEvents()
#         for i in range(20):
#             action = np.array([0, 0, -1])
#             state, done = env.collect_demo(buffer, env, state, action)
#             step += 1
#             if step >= buffer_size:
#                 print('Collect push done')
#                 buffer.save(os.path.join(
#                             'buffers',
#                             'panda-grasp',
#                             f'ur10-v3-00928.pth'
#                         ))
#         for i in range(10):
#             action = np.array([0, 0, 1])
#             state, done = env.collect_demo(buffer, env, state, action)
#             step += 1
#             if step >= buffer_size:
#                         print('Collect reach done')
#                         buffer.save(os.path.join(
#                             'buffers',
#                             'panda-stack',
#                             f'ur10-v3-00928.pth'
#                         ))
#         for i in range(20):
#             action = np.array([0, 1, 0])
#             state, done = env.collect_demo(buffer, env, state, action)
#             step += 1
#             if step >= buffer_size:
#                         print('Collect reach done')
#                         buffer.save(os.path.join(
#                             'buffers',
#                             'panda-stack',
#                             f'ur10-v3-00928.pth'
#                         ))
#             if done:
#                 env.reset()
    # while True:
    #     p.stepSimulation()
    #     keys = p.getKeyboardEvents()
    #     if env.infomation():
    #         if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([0, -1, 0])
    #
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([0, 1, 0])
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([1, 0, 0])
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([-1, 0, 0])
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP] & p.KEY_IS_DOWN:
    #             action = np.array([0, 0, 1])
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_PAGE_DOWN in keys and keys[p.B3G_PAGE_DOWN] & p.KEY_IS_DOWN:
    #             action = np.array([0, 0, -1])
    #             state, done = env.collect_demo(buffer, env, state, action)
    #             step1 += 1
    #             if done:
    #                 env.reset()
    #
    #         if step >= buffer_size:
    #             print('Collect push done')
    #             buffer.save(os.path.join(
    #                 'buffers',
    #                 'ur10-push',
    #                 f'ur10-v3-00926.pth'
    #             ))
    #     else:
    #         if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([0, -1, 0])
    #
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #
    #                 env.reset()
    #
    #
    #         if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([0, 1, 0])
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([1, 0, 0])
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #         if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
    #             action = np.array([-1, 0, 0])
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #
    #             if done:
    #                 env.reset()
    #
    #
    #         if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP] & p.KEY_IS_DOWN:
    #             action = np.array([0, 0, 1])
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #             if done:
    #                 env.reset()
    #
    #
    #         if p.B3G_PAGE_DOWN in keys and keys[p.B3G_PAGE_DOWN] & p.KEY_IS_DOWN:
    #             action = np.array([0, 0, -1])
    #             state, done = env.collect_demo(buffer1, env, state, action)
    #             step1 += 1
    #             if done:
    #                 env.reset()
    #
    #
    #
    #         if step1 >= buffer_size1:
    #             print('Collect reach done')
    #             buffer1.save(os.path.join(
    #                 'buffers',
    #                 'ur10-reach',
    #                 f'ur10-v3-00926.pth'
    #             ))
    #
    #
    #     if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
    #         action = np.array([0, -1, 0])
    #         if stage == 0:
    #             state, done, stage = env.collect_demo(buffer, env, state, action)
    #
    #
    #         else:
    #             state, done, stage = env.collect_demo(buffer1, env, state, action)
    #
    #         if done:
    #             env.reset()
    #             step += 1
    #
    #
    #     if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
    #         action = np.array([0, 1, 0])
    #
    #         if stage == 0:
    #             state, done, stage = env.collect_demo(buffer, env, state, action)
    #
    #
    #         else:
    #             state, done, stage = env.collect_demo(buffer1, env, state, action)
    #
    #         if done:
    #             env.reset()
    #             step += 1
    #
    #
    #
    #     if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP] & p.KEY_IS_DOWN:
    #         action = np.array([-1, 0, 0])
    #         state, done, stage = env.collect_demo(buffer1, env, state, action)
    #
    #         if done:
    #             env.reset()
    #             step += 1
    #
    #
    #     if p.B3G_PAGE_DOWN in keys and keys[p.B3G_PAGE_DOWN] & p.KEY_IS_DOWN:
    #         action = np.array([1, 0, 0])
    #         state, done, stage = env.collect_demo(buffer1, env, state, action)
    #
    #         if done:
    #             env.reset()
    #             step += 1






# first stage is push second stage is reach
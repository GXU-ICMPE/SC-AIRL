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
        self.x_low_obs = 0.2
        self.x_high_obs = 0.4
        self.y_low_obs = -0.04
        self.y_high_obs = 1.2
        self.z_low_obs = 0.03
        self.z_high_obs = 0.3

        self.target_x_low_obs = 0.2
        self.target_x_high_obs = 0.4
        self.target_y_low_obs = -0.04
        self.target_y_high_obs = 1.14
        self.target_z_low_obs = 0.03
        self.target_z_high_obs = 0.2

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

        self.observation_space = spaces.Box(
            low=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_low_obs, self.target_y_low_obs, self.target_z_low_obs, -1.0
                 ],
                dtype=float),
            high=np.array(
                [self.x_low_obs, self.y_low_obs, self.z_low_obs,
                 self.target_x_high_obs, self.target_y_high_obs, self.target_z_high_obs, 1.0
                 ]),
            dtype=float
        )

        self.z = 0
        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1
        self.G = 0


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
        self.file_path = "D:\\pycharm_project\\xiangguangyu_SC_AIRL\\UR5-AIRL-master\\result\\output.txt"
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

        self.step_counter = 0
        self.stage = 0
        self.done = False
        self.done1 = False
        self.G = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.finish_task_one = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.loadURDF("plane.urdf", basePosition=[0, 0, 0])

        # p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
        # p.loadURDF("tray/tray.urdf", basePosition=[0.62, 0, 0.0])
        # p.loadURDF("tray/traybox.urdf", [0.62, 0, 0.0])
        # self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.64, 0.0, 0.02])

        # self.cube2 = p.loadURDF("cube4.urdf", basePosition=[0.6, 0.07, 0.0], useFixedBase=True)
        self.cube1 = p.loadURDF("cube2.urdf", basePosition=[0.3, 1.0, 0.02], useFixedBase=True)
        #
        # self.random_x = random.uniform(0.52,0.525)
        # self.random_y = random.uniform(-0.05, -0.045)
        #
        # self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.6, -0.01, 0.02])
        self.random_x = random.uniform(0.62, 0.58)
        self.random_y = random.uniform(-0.02, -0.01)

        p.loadURDF("cube.urdf", basePosition=[0.3, -0.007, 0.02], useFixedBase=True)
        self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.3, -0.0, 0.06])

        # self.random_number = random.randint(1, 3)
        # if self.random_number == 1:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.62, -0.02, 0.02])
        # if self.random_number == 2:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.6, -0.02, 0.02])
        # if self.random_number == 3:
        #     self.target_cube = p.loadURDF("cube.urdf", basePosition=[0.64, -0.02, 0.02])


        # self.UR5 = p.loadURDF("D:\\pycharm_project\\untitled\\ur10urdf\\ur10.urdf", useFixedBase=True)
        self.UR5 = p.loadURDF("franka_panda/panda.urdf", basePosition=np.array([0, 0, 0]), useFixedBase=True)

        self.UR52 = p.loadURDF("franka_panda/panda.urdf", basePosition=np.array([0, 1.0, 0]), useFixedBase=True)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[0.0, 0.5, 0.0]
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
            p.resetJointState(self.UR52, j, rest_poses[j])

        p.resetJointState(self.UR5, 9, 0.08)
        p.resetJointState(self.UR5, 10, 0.08)
        x = random.uniform(-0.05, 0.05)
        y = random.uniform(-0.05, 0.05)
        z = random.uniform(0.0, 0.05)
        init_position = [0.3, 0.0, 0.2]  # train position
        # init_position = [0.5308569, 0.08824733, 0.40342805]     # after train position
        init_postrue_vary = [0.0, -1.57079632, 0.0]

        init_posture = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)

        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.0)  # 初始姿态偏置
        jointposes = p.calculateInverseKinematics(self.UR5, 11, position_start, postrue_start, maxNumIterations=100)
        for j in range(7):
            p.resetJointState(self.UR5, j, jointposes[j])


        p.resetJointState(self.UR52, 9, 0.08)
        p.resetJointState(self.UR52, 10, 0.08)

        init_position = [0.3, 0.7, 0.2]  # train position
        # init_position = [0.5308569, 0.08824733, 0.40342805]     # after train position
        init_postrue_vary = [0.0, 0.0, 0.0]

        init_posture = np.array(p.getLinkState(self.UR52, 11)[5], dtype=float).reshape(4, 1)

        position_start, postrue_start, res = self.get_position_r_vary(init_position, init_posture, E=init_postrue_vary,
                                                                      x=0.036)  # 初始姿态偏置
        jointposes = p.calculateInverseKinematics(self.UR52, 11, position_start, postrue_start, maxNumIterations=100)
        for j in range(7):
            p.resetJointState(self.UR52, j, jointposes[j])




        self.postrue = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
        self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)




        # for _ in range(90):
        #     self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
        #     self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        #     self.position_target = self.position + np.array([0.0, 0.0, 0.001]).reshape(3, 1)
        #     # self.postrue = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
        #     self.move(self.position_target, self.postrue)
        #     position2 = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)
        #     postrue2 = np.array(p.getLinkState(self.UR52, 11)[5], dtype=float).reshape(4, 1)
        #     self.move2(position2, postrue2)

        self.position1 = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        self.postrue1 = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)

        self.position2 = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)
        self.postrue2 = np.array(p.getLinkState(self.UR52, 11)[5], dtype=float).reshape(4, 1)

        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)
        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)


        self.observation_init = np.array(
            [self.position1[0][0], self.position1[1][0], self.position1[2][0],
             self.position_target_cube[0][0], self.position_target_cube[0][1], self.position_target_cube[0][2], self.stage],
            dtype=float)

        self.c = 0

        return self.observation_init
    #
    # def collect_ter_set(self, state):



    def step(self, action):

        # if self.stage == 1 and self.G == 0:
        #     self.stage = 0
        self.ua = 0
        self.ub = 0
        self.uc = 0
        self.ud = 0
        if self.stage == 0 or self.stage == 1:
            self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        else:
            self.position = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)

        dx = 0.005 * action[0]
        dy = 0.005 * action[1]
        dz = 0.005 * action[2]

        self.position_target = self.position + np.array([dx, dy, dz]).reshape(3, 1)
        # self.postrue = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
        if self.stage == 0 or self.stage == 1:
            self.move(self.position_target, self.postrue)
            self.move2(self.position2, self.postrue2)
            self.move_gripper2(self.position2, self.postrue2, list([0.04, 0.04]))
        elif self.stage == 2:
            self.move2(self.position_target, self.postrue)
            self.move(self.position1, self.postrue1)
            self.move_gripper(self.position1, self.postrue1, list([0.018, 0.018]))
        elif self.stage == 3:
            self.move_gripper(self.position1, self.postrue1, list([0.04, 0.04]))
            self.move2(self.position_target, self.postrue)
            self.move(self.position1, self.postrue1)

        if self.stage == 0 or self.stage == 1:
            self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        else:
            self.position = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)

        self.position_cube1 = p.getBasePositionAndOrientation(self.cube1)
        self.position_target_cube = p.getBasePositionAndOrientation(self.target_cube)

        if self.stage == 0 or self.stage == 1:
            self.distance = sqrt((self.position[0][0]-self.position_target_cube[0][0])**2 + (self.position[1][0] - self.position_target_cube[0][1] + 0.012)**2 + (self.position[2][0]-self.position_target_cube[0][2]+0.005)**2)

            self.distance1 = sqrt((self.position_target_cube[0][0] - 0.3) ** 2 + (self.position_target_cube[0][1] - 0.5) ** 2 + (self.position_target_cube[0][2] - 0.15) ** 2)
        else:
            self.distance = sqrt((self.position[0][0] - self.position_target_cube[0][0]) ** 2 + (self.position[1][0] - self.position_target_cube[0][1] - 0.01) ** 2 + (self.position[2][0] - self.position_target_cube[0][2] - 0.01) ** 2)

            self.distance1 = sqrt((self.position_target_cube[0][0] - 0.3) ** 2 + (self.position_target_cube[0][1] - 1.0) ** 2) # + (self.position_target_cube[0][2] - 0.06) ** 2)



        if self.stage == 0:
            print(action)
            self.move_gripper(self.position, self.postrue, list([0.04, 0.04]))
            if self.distance <= 0.01:
                self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
                self.a = 1
            else:
                self.contact = p.getContactPoints(self.UR5, self.target_cube)
                if self.contact:
                    self.ua = 1
        elif self.stage == 1:
            print(action)
            self.move_gripper(self.position, self.postrue, list([0.015, 0.015]))
            if self.distance1 <= 0.01:
                self.b = 1

            else:
                self.contact = p.getContactPoints(self.UR5, self.target_cube)
                if not self.contact:
                    self.ub = 1
        elif self.stage == 2:
            print(action)
            self.move_gripper2(self.position, self.postrue, list([0.04, 0.04]))
            if self.distance <= 0.01:
                self.move_gripper2(self.position, self.postrue, list([0.018, 0.018]))
                self.c = 1
            else:
                self.contact = p.getContactPoints(self.UR52, self.target_cube)
                if self.contact:
                    self.uc = 1
        elif self.stage == 3:
            print(action)
            self.move_gripper2(self.position, self.postrue, list([0.04, 0.04]))
            self.move_gripper2(self.position, self.postrue, list([0.018, 0.018]))
            if self.distance1 <= 0.009:
                self.d = 1
                self.move_gripper2(self.position, self.postrue, list([0.04, 0.04]))
            else:
                self.contact = p.getContactPoints(self.UR52, self.UR5)
                self.contact1 = p.getContactPoints(self.target_cube, self.UR5)
                if self.contact or self.contact1:
                    self.ud = 1
        self.step_counter += 1
        n = 1000
        while (n):
            p.stepSimulation()
            n = n - 1

        info = self.infomation()

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

    def move2(self, position, postrue):

        jointposes = p.calculateInverseKinematics(self.UR52, 11, position, postrue, maxNumIterations=100)
        p.setJointMotorControlArray(self.UR52, list([0,1,2,3,4,5,6,9,10]), p.POSITION_CONTROL
                                    , list(jointposes))


    def move_gripper(self, position, postrue, move):

        p.setJointMotorControlArray(self.UR5, list([9,10]), p.POSITION_CONTROL, move)


    def move_gripper2(self, position, postrue, move):

        p.setJointMotorControlArray(self.UR52, list([9,10]), p.POSITION_CONTROL, move)


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
        if self.stage == 0 or self.stage == 1:
            self.position = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
        else:
            self.position = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)
        terminated = bool(
                self.position[0][0] < self.x_low_obs
                or self.position[0][0] > self.x_high_obs
                or self.position[1][0] < self.y_low_obs
                or self.position[1][0] > self.y_high_obs
                or self.position[2][0] < self.z_low_obs
                or self.position[2][0] > self.z_high_obs
            )

        self.reward = 0

        self.contact1 = p.getContactPoints(self.target_cube, self.cube1)
        # if terminated:
        #     self.reward = float(-100)
        #     self.done = True

        if self.step_counter >= self._max_episode_steps:
            self.reward = float(-100)
            self.done = True

        elif self.stage == 0 and self.a == 0:
            self.reward = -self.distance
            self.done = False

        elif self.stage == 0 and self.ua == 1:
            self.reward = -0.1 - self.distance
            self.done = False

        elif self.stage == 0 and self.a == 1:
            self.reward = 1
            self.stage = 1
            self.done = False
            print(111)

        elif self.stage == 1 and self.b == 0:
            self.reward = -self.distance1
            self.done = False

        elif self.stage == 1 and self.ub == 1:
            self.reward = -1 - self.distance1
            self.done = True

        elif self.stage == 1 and self.b == 1:
            self.position1 = np.array(p.getLinkState(self.UR5, 11)[4], dtype=float).reshape(3, 1)
            self.postrue1 = np.array(p.getLinkState(self.UR5, 11)[5], dtype=float).reshape(4, 1)
            self.stage = 2
            self.position = np.array(p.getLinkState(self.UR52, 11)[4], dtype=float).reshape(3, 1)
            self.postrue = np.array(p.getLinkState(self.UR52, 11)[5], dtype=float).reshape(4, 1)
            self.done = False
            print(222)

        elif self.stage == 2 and self.c == 0 and self.uc == 0:
            self.reward = -self.distance
            self.done = False

        elif self.stage == 2 and self.c == 0 and self.uc == 1:
            self.reward = -1 - self.distance
            self.done = False

        elif self.stage == 2 and self.c == 1:
            self.reward = 10
            self.stage = 3
            self.done = False
            print(333)

        elif self.stage == 3 and self.d == 0:
            self.reward = -self.distance1
            self.done = False

        elif self.stage == 3 and self.d == 0 and self.ud == 1:
            self.reward = -1 - self.distance1
            self.done = False

        elif self.stage == 3 and self.d == 1:
            self.reward = 10
            self.done = True
            print(444)


        if self.stage == 0:
            self.observation = np.array(
                [self.position[0][0], self.position[1][0], self.position[2][0],
                 self.position_target_cube[0][0], self.position_target_cube[0][1],
                 self.position_target_cube[0][2], self.stage],
                dtype=float)
        elif self.stage == 1:
            self.observation = np.array(
                [self.position_target_cube[0][0], self.position_target_cube[0][1],
                 self.position_target_cube[0][2], 0.3, 0.5, 0.15, self.stage],
                dtype=float)
        elif self.stage == 2:

            self.observation = np.array(
                [self.position[0][0], self.position[1][0], self.position[2][0],
                 self.position_target_cube[0][0], self.position_target_cube[0][1],
                 self.position_target_cube[0][2], self.stage],
                dtype=float)
        elif self.stage == 3:
            self.observation = np.array(
                [self.position_target_cube[0][0], self.position_target_cube[0][1],
                self.position_target_cube[0][2], 0.3, 1.0, 0.06, self.stage],
                dtype=float)

        info = self.infomation()

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
        env.step([0.0, 0.0, 0.0])
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
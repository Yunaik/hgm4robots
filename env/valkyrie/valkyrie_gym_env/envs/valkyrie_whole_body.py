from valkyrie_gym_env.envs.loadObstacles import loadPebble, loadV, loadSlab, loadStep, loadPlank, loadSeesaw1, loadSeesaw2, loadSlipPlate
from valkyrie_gym_env.envs.minimum_jerk import mjtg
from softlearning.policies.utils import get_policy_from_variant
from softlearning.environments.utils import get_environment_from_params
import pickle
import json
import tensorflow as tf
from gym import spaces
import copy
import numpy as np
from .valkyrie import Valkyrie
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getPolicy(checkpoint_path=None):
    checkpoint_path = os.path.dirname(__file__)\
        + "/nn_policies/policy1/checkpoint_6001"\
        if checkpoint_path is None else checkpoint_path
    experiment_path = os.path.dirname(checkpoint_path)
    variant_path = os.path.join(experiment_path, 'params.json')

    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with tf.keras.backend.get_session().as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    evaluation_environment = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    return policy


class Valkyrie_whole_body(Valkyrie):
    def __init__(self,
                 reach_weight_dic=None,
                 upper_policy=None,
                 spawn_objects=False,
                 goal_type="fixed",
                 control_mode="whole_body",  # lower, upper, whole_body
                 incremental_control_high=False,
                 random_spawn=False,
                 load_obstacle=False,
                 high_level_frequency=0.5,
                 reward_pose_penalty=1,
                 reward_box_in_hand=1,
                 reward_box_drop=2,
                 reward_door_is_open=2,
                 reward_is_at_goal=5,
                 debug=True,
                 useMinimumJerk=False,
                 *args, **kwargs):
        assert control_mode != "upper" or goal_type != "random_fixed"
        assert control_mode == "whole_body", "Only whole body support. use Valkyrie class for lower and upper body"
        self.useMinimumJerk = useMinimumJerk

        self.load_obstacle = load_obstacle
        self.obstacle_frames = []
        self.obstacle_links_to_read = []
        self.incremental_control_high = incremental_control_high
        self.spawn_objects = spawn_objects
        self.is_at_goal = False
        self.control_mode = control_mode
        self.arm_counter = 0
        self.links_to_read = None
        self.lower_policy_side = getPolicy(os.path.dirname(
            __file__)+"/nn_policies/policy2/checkpoint_9000")

        self.lower_policy_straight = getPolicy()
        self.upper_policy = upper_policy
        lower_body_obs_dim = (11
                              + 2  # imitate_motion
                              + 1  # *self.obs_use_yaw # yaw of pelvis
                              )
        upper_body_obs_dim = 6+6+6+100  # sort out
        self.reward_pose_penalty = reward_pose_penalty
        self.reward_box_in_hand = reward_box_in_hand
        self.reward_box_drop = reward_box_drop
        self.reward_door_is_open = reward_door_is_open
        self.reward_is_at_goal = reward_is_at_goal

        q_nom = dict([
            ("rightHipYaw", 0.0),
            ("rightHipRoll", -0.1),
            ("rightHipPitch",    -0.45*1.),
            ("rightKneePitch",   0.944*1.),
            ("rightAnklePitch",   -0.527*1.),
            ("rightAnkleRoll", 0.1),
            ("leftHipYaw", 0.0),
            ("leftHipRoll", 0.1),
            ("leftHipPitch",     -0.45*1.),
            ("leftKneePitch",    0.944*1.),
            ("leftAnklePitch",    -0.527*1.),
            ("leftAnkleRoll", -0.1),
            ("torsoYaw", 0.0),
            ("torsoPitch", 0.0),
            ("torsoRoll", 0.0),
            ("rightShoulderPitch", 0.300196631343),
            ("rightShoulderRoll", 0.65),
            ("rightShoulderYaw", 0.0),
            ("rightElbowPitch", 1.785398163397),
            ("leftShoulderPitch", 0.300196631343),
            ("leftShoulderRoll", -0.65),
            ("leftShoulderYaw", 0.0),
            ("leftElbowPitch", -1.785398163397),
        ])

        super(Valkyrie_whole_body, self).__init__(
            useFullDOF=True,
            start_index=upper_body_obs_dim+lower_body_obs_dim,
            q_nom=q_nom,
            goal_type=goal_type,
            whole_body_mode=True,
            not_reaching=control_mode != "upper",
            random_spawn=random_spawn,
            debug=False,
            *args, **kwargs)
        self.debug = debug

        self.high_level_frequency = high_level_frequency
        assert self.PD_freq % self.high_level_frequency == 0
        self.high_level_repeat = int(self.PD_freq/self.high_level_frequency)
        """Weight dic"""

        self.reach_weight_dic = {
            "weight_lx_pos_reward":         1.0/3.0,
            "weight_ly_pos_reward":         1.0/3.0,
            "weight_lz_pos_reward":         1.0/3.0,
            "weight_rx_pos_reward":         1.0/3.0,
            "weight_ry_pos_reward":         1.0/3.0,
            "weight_rz_pos_reward":         1.0/3.0,
            "weight_lx_vel_reward":         1.0/6.0,
            "weight_ly_vel_reward":         1.0/6.0,
            "weight_lz_vel_reward":         1.0/6.0,
            "weight_rx_vel_reward":         1.0/6.0,
            "weight_ry_vel_reward":         1.0/6.0,
            "weight_rz_vel_reward":         1.0/6.0,
            "weight_joint_vel_reward":      0.5,
            "weight_joint_torque_reward":   0.5,

            # Clamp reward
            "weight_box_pos_x_reward":      1.0/3.0,
            "weight_box_pos_y_reward":      1.0/3.0,
            "weight_box_pos_z_reward":      1.0/3.0,
            "weight_contact_reward":        1.0,
            "weight_box_gravity_reward":    1.0,
            "weight_box_vel_x_reward":      1.0/6.0,
            "weight_box_vel_y_reward":      1.0/6.0,
            "weight_box_vel_z_reward":      1.0/6.0,
        } if reach_weight_dic is None else reach_weight_dic

        """PD gains"""

        kp_shoulder_pitch = 1089
        kp_shoulder_roll = 1089
        kp_shoulder_yaw = 372
        kp_elbow_pitch = 1000

        kd_shoulder_pitch = 10.9
        kd_shoulder_roll = 10.9
        kd_shoulder_yaw = 1
        kd_elbow_pitch = 2

        kp_unit = 10
        kd_unit = 0.12

        kp_hip_pitch = 1000
        kp_knee_pitch = 1000
        kp_ankle_pitch = 1000

        kd_hip_pitch = 5
        kd_knee_pitch = 10
        kd_ankle_pitch = 1

        self.Kp = dict([
            ("torsoYaw",    190 * kp_unit),
            ("torsoPitch",  150 * kp_unit),
            ("torsoRoll",   150 * kp_unit),

            ("rightShoulderPitch",  kp_shoulder_pitch),
            ("rightShoulderRoll",   kp_shoulder_roll),
            ("rightShoulderYaw",    kp_shoulder_yaw),
            ("rightElbowPitch",     kp_elbow_pitch),
            ("leftShoulderPitch",   kp_shoulder_pitch),
            ("leftShoulderRoll",    kp_shoulder_roll),
            ("leftShoulderYaw",     kp_shoulder_yaw),
            ("leftElbowPitch",      kp_elbow_pitch),

            ("rightHipYaw",     190 * kp_unit),
            ("rightHipRoll",    350 * kp_unit),
            ("rightHipPitch",   kp_hip_pitch),
            ("rightKneePitch",  kp_knee_pitch),
            ("rightAnklePitch", kp_ankle_pitch),
            ("leftHipYaw",      190 * kp_unit),
            ("leftHipRoll",     350 * kp_unit),
            ("leftHipPitch",    kp_hip_pitch),
            ("leftKneePitch",   kp_knee_pitch),
            ("leftAnklePitch",  kp_ankle_pitch),
        ])

        self.Kd = dict([
            ("torsoYaw", 190 * kd_unit),
            ("torsoPitch", 150 * kd_unit),
            ("torsoRoll", 150 * kd_unit),

            ("rightShoulderPitch",  kd_shoulder_pitch),
            ("rightShoulderRoll",   kd_shoulder_roll),
            ("rightShoulderYaw",    kd_shoulder_yaw),
            ("rightElbowPitch",     kd_elbow_pitch),
            ("leftShoulderPitch",   kd_shoulder_pitch),
            ("leftShoulderRoll",    kd_shoulder_roll),
            ("leftShoulderYaw",     kd_shoulder_yaw),
            ("leftElbowPitch",      kd_elbow_pitch),

            ("rightHipYaw",     190 * kd_unit),
            ("rightHipRoll",    350 * kd_unit),
            ("rightHipPitch",   kd_hip_pitch),
            ("rightKneePitch",  kd_knee_pitch),
            ("rightAnklePitch", kd_ankle_pitch),
            ("leftHipYaw",      190 * kd_unit),
            ("leftHipRoll",     350 * kd_unit),
            ("leftHipPitch",    kd_hip_pitch),
            ("leftKneePitch",   kd_knee_pitch),
            ("leftAnklePitch",  kd_ankle_pitch),
        ])
        self.lower_controlled_joints = [
            "rightHipRoll",
            "rightHipPitch",
            "rightKneePitch",
            "rightAnklePitch",
            "leftHipRoll",
            "leftHipPitch",
            "leftKneePitch",
            "leftAnklePitch",
        ]

        self.upper_controlled_joints = [
            "rightShoulderPitch",
            "rightShoulderRoll",
            "rightShoulderYaw",
            "rightElbowPitch",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "leftElbowPitch"]

        """Copy pasted from reach"""
        # Some flags
        self.initialised = False

        self.arm_action = 0

        if self.control_mode == "upper":
            # hardcoding here
            self._actionDim = 8
            self._observationDim = upper_body_obs_dim+self._actionDim
            jointLowerLimit = []
            jointUpperLimit = []

            for jointName in self.upper_controlled_joints:
                jointLowerLimit.append(self.joint_limits_low[jointName])
                jointUpperLimit.append(self.joint_limits_high[jointName])

            joint_increment = (np.array(jointUpperLimit) -
                               np.array(jointLowerLimit))/1

            self.action_space = spaces.Box(-np.array(joint_increment),
                                           np.array(joint_increment))

            observation_high = np.array(
                [np.finfo(np.float32).max] * self._observationDim)
            self.observation_space = spaces.Box(
                -observation_high, observation_high)

        elif self.control_mode == "lower":
            self._actionDim = 8
            self._observationDim = lower_body_obs_dim+self._actionDim
            jointLowerLimit = []
            jointUpperLimit = []

            for jointName in self.lower_controlled_joints:
                jointLowerLimit.append(self.joint_limits_low[jointName])
                jointUpperLimit.append(self.joint_limits_high[jointName])

            joint_increment = (np.array(jointUpperLimit) -
                               np.array(jointLowerLimit))/1

            self.action_space = spaces.Box(-np.array(joint_increment),
                                           np.array(joint_increment))

            observation_high = np.array(
                [np.finfo(np.float32).max] * self._observationDim)
            self.observation_space = spaces.Box(
                -observation_high, observation_high)

        elif self.control_mode == "whole_body":

            box_offset = 0.5

            goals = [[2.5+box_offset, -1], [4.5+box_offset, -2], [7.0, -2]]
            reach_arm_states = [0, 1, 2]
            stop_gaits = [0]

            self.action_combination = []
            action_idx = 0
            for goal in goals:
                for reach_arm_state in reach_arm_states:
                    for stop_gait in stop_gaits:
                        self.action_combination.append(
                            [goal, reach_arm_state, stop_gait])
                        if self.debug:
                            print("Action %d corresponds to pelvis goal: [%.2f, %.2f], reach arm: %d, stop gait: %d"
                                  % (action_idx, goal[0], goal[1], reach_arm_state, stop_gait))
                        action_idx += 1
            self._actionDim = len(self.action_combination)
            self.action_space = spaces.Discrete(self._actionDim)
            if self.debug:
                print("Reach arms 0: nominal, 1: door, 2: box")
            self._observationDim = 2+2+1+1+1+1+1
            observation_high = np.array(
                [np.finfo(np.float32).max] * self._observationDim)
            self.observation_space = spaces.Box(
                -observation_high, observation_high)
        else:
            print("Control mode %s does not exist " % self.control_mode)

        self.joint_limits_absolute_low = {}
        self.joint_limits_absolute_high = {}
        print("Action Dim: %d, Observation Dim: %d" %
              (self._actionDim, self._observationDim))
        tolerance = 0.07
        for joint in self.controlled_joints:
            info = self._p.getJointInfo(self.robot, self.jointIdx[joint])
            self.joint_limits_absolute_low.update({joint: (info[8])})
            self.joint_limits_absolute_high.update({joint: (info[9])})
            assert self.joint_limits_absolute_low[joint] - \
                tolerance <= self.joint_limits_low[joint], "Lower limits have to lower than imitation weights"
            assert self.joint_limits_absolute_high[joint] + \
                tolerance >= self.joint_limits_high[joint], "Upper limits have to greater than imitation weights"
        self.jointLowerLimitAbsolute = []
        self.jointUpperLimitAbsolute = []

        for jointName in self.controlled_joints:
            self.jointLowerLimitAbsolute.append(
                self.joint_limits_absolute_low[jointName])
            self.jointUpperLimitAbsolute.append(
                self.joint_limits_absolute_high[jointName])

        self.is_at_middle_goal = False
        self.box_in_hand = False
        self.middle_goal = [self.table2_pos[0]-0.75, self.table2_pos[1], 1.175]
        self.end_goal = [7.0, -2.0, 1.175]

        if self.visualise_goal:
            self.end_goal_object = self._p.loadURDF(
                self.dir_path + "/urdf/end_target.urdf",
                basePosition=self.end_goal,
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True)

        """ARM IK"""

        # Set up and calculate trajectory.
        self.arm_time = 2.0
        self.arm_frequency = 25
        self.arm_idx = 0

        self._left_target = [0, 0, 0]
        self._right_target = [0, 0, 0]

    def spawn_obstacle(self):
        if self.load_obstacle == 1:
            box_pos1 = [4., -1.7, 0.2]
            self.box_obstacle1 = self._p.loadURDF(self.dir_path + "/urdf/box_obstacle.urdf",
                                                  basePosition=box_pos1,
                                                  baseOrientation=[0.0, 0.0, 0.707107, 0.707107])

            box_pos2 = [4.8, -2.4, 0.2]
            self.box_obstacle2 = self._p.loadURDF(self.dir_path + "/urdf/box_obstacle.urdf",
                                                  basePosition=box_pos2,
                                                  baseOrientation=[0.0, 0.0, 0.707107, 0.707107])

            box_pos3 = [4.5, -1.5, 0.2]
            self.box_obstacle3 = self._p.loadURDF(self.dir_path + "/urdf/box_obstacle.urdf",
                                                  basePosition=box_pos3,
                                                  baseOrientation=[0.0, 0.0, 0.707107, 0.707107])
        elif self.load_obstacle == 2:
            self.obstacle4, self.obstacle5 = loadV(self._p)
        elif self.load_obstacle == 3:
            self.obstacle4 = loadSlipPlate(
                self._p, pos_x=4.0, pos_y=-2.0, lateral_friction=0.2)
            self.obstacle5 = loadSlipPlate(
                self._p, pos_x=2.0, pos_y=-1.0, lateral_friction=0.4)

    def high_level_reward(self):
        reward = 0.1  # survival bonus
        reward_info = {}

        # reward for being at middle goal
        if self.is_at_middle_goal:
            reward += 1
            reward -= (not self.upper_joints_are_nominal) * \
                self.reward_pose_penalty
            reward += self.reward_box_drop if (
                self.box_on_table and not self.box_in_hand) else -self.reward_box_drop
        else:
            reward -= (self.upper_joints_are_nominal)*self.reward_pose_penalty
            reward += self.reward_box_in_hand if self.box_in_hand else -self.reward_box_in_hand

        reward += self.door_is_open*self.reward_door_is_open
        # reward for being at final goal
        reward += self.is_at_goal*self.reward_is_at_goal
        reward_info.update({
            "is_at_middle_goal": self.is_at_middle_goal,
            "upper_joints_are_nominal": self.upper_joints_are_nominal,
            "box_in_hand": self.box_in_hand,
            "door_is_open": self.door_is_open,
            "is_at_goal": self.is_at_goal,
            "box_on_table": self.box_on_table,
        })

        return reward, reward_info

    def getReward(self):

        if self.control_mode == "upper":
            reach_reward, reach_reward_info = self.reach_reward()  # reward for clamp/reach

            reward_dict = {}
            reward_dict.update(reach_reward_info)
            reward = reach_reward
        elif self.control_mode == "lower":
            walk_reward, walk_reward_info = super().getReward()    # imitation and goal

            reward_dict = {}
            reward_dict.update(walk_reward_info)
            reward = walk_reward
        elif self.control_mode == "whole_body":
            high_level_reward, high_level_reward_info = self.high_level_reward()

            reward_dict = {}
            reward_dict.update(high_level_reward_info)

            reward = high_level_reward
        else:
            print("Control mode %s does not exist " % self.control_mode)

        return reward, reward_dict

    def reset(self, start_frame=0, isTest=False):
        self.is_at_goal = False
        self.arm_counter = 0
        self.box_hold_counter = 0
        self.is_at_middle_goal = False
        self.box_in_hand = False
        self.box_on_table = False
        self.door_is_open = False
        self.lower_policy_straight.reset()
        self.lower_policy_side.reset()

        if self.upper_policy is not None:
            self.upper_policy.reset()

        self.initialised = False

        super().reset(loadObject=False, start_frame=start_frame, isTest=isTest)
        if self.exertForce:
            print("Disturbance time: ", self.disturbance_time)
        if self.load_obstacle:
            self.spawn_obstacle()

        self.pelvis_goal = [1., 0, 1.175]

        if self.spawn_objects:
            self.loadAllObjects()
        if self.visualise_goal:
            self.end_goal_object = self._p.loadURDF(
                self.dir_path + "/urdf/end_target.urdf",
                basePosition=self.end_goal,
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True)
        self.time_last_move_goal = -1
        self.initialised = True

        obs = self.get_observation()
        self.obstacle_ids = []
        self.obstacle_ids.append(self.door)
        self.obstacle_ids.append(self.table)
        self.obstacle_ids.append(self.table_final)
        self.obstacle_ids.append(self.box)
        if self.load_obstacle == 1:
            self.obstacle_ids.append(self.box_obstacle1)
            self.obstacle_ids.append(self.box_obstacle2)
            self.obstacle_ids.append(self.box_obstacle3)
        elif self.load_obstacle == 2 or self.load_obstacle == 3:
            self.obstacle_ids.append(self.obstacle4)
            self.obstacle_ids.append(self.obstacle5)
        return copy.copy(obs)

    def changePDgains(self, pd_set=1):

        if pd_set == 1:
            kp_hip_pitch = 2000
            kp_knee_pitch = 2000
            kp_ankle_pitch = 1500
            kd_hip_pitch = 10
            kd_knee_pitch = 10
            kd_ankle_pitch = 1
        else:
            kp_hip_pitch = 1000
            kp_knee_pitch = 1000
            kp_ankle_pitch = 1000

            kd_hip_pitch = 5
            kd_knee_pitch = 10
            kd_ankle_pitch = 1

        self.Kp.update(dict([
            ("rightHipPitch",   kp_hip_pitch),
            ("rightKneePitch",  kp_knee_pitch),
            ("rightAnklePitch", kp_ankle_pitch),
            ("leftHipPitch",    kp_hip_pitch),
            ("leftKneePitch",   kp_knee_pitch),
            ("leftAnklePitch",  kp_ankle_pitch),
        ]))

        self.Kd.update(dict([
            ("rightHipPitch",   kd_hip_pitch),
            ("rightKneePitch",  kd_knee_pitch),
            ("rightAnklePitch", kd_ankle_pitch),
            ("leftHipPitch",    kd_hip_pitch),
            ("leftKneePitch",   kd_knee_pitch),
            ("leftAnklePitch",  kd_ankle_pitch),
        ])
        )

    def get_observation(self):
        self.reach_obs = np.array(
            self.getExtendedObservation_reach()) if self.spawn_objects else 0
        self.walk_obs = np.array(self.getExtendedObservation())

        """Box stuff"""
        self.box_on_table = len(self._p.getContactPoints(self.table_final,
                                                         self.box, -1, -1)) > 0 and self.box_pos[2] > 0.97\
            or self.box_on_table if self.initialised else False

        if self.control_mode == "upper":
            assert self.reach_obs.shape[0] == self._observationDim
            return copy.copy(self.reach_obs)
        elif self.control_mode == "lower":
            assert self.walk_obs.shape[0] == self._observationDim
            return copy.copy(self.walk_obs)
        elif self.control_mode == "whole_body":
            # make sure no element is double
            # goal and base pos and upper joint position
            observation = []
            observation.append((self.middle_goal[0]-self.base_pos[0])/7)
            observation.append((self.middle_goal[1]-self.base_pos[1])/2)
            observation.append((self.end_goal[0]-self.base_pos[0])/7)
            observation.append((self.end_goal[1]-self.base_pos[1])/2)

            # normalise, box shouldnt be further away that 1.7m
            observation.append((self.middle_goal[0]-self.box_pos[0])/1.7)

            observation.append(self.box_in_hand)
            observation.append(self.box_on_table)
            observation.append(self.door_is_open)
            self.upper_joints_are_nominal = copy.copy(self.arm_action == 0)
            observation.append(self.upper_joints_are_nominal)

            self.obs = copy.copy(np.array(observation))
            return copy.copy(np.array(observation))
        else:
            raise ValueError("Mode does not exist")

    def getHandSetPoint(self, left_hand_goal, right_hand_goal):
        """Offline planning version"""
        if self.useMinimumJerk:
            if self.arm_idx == 0:
                self.left_traj = mjtg(
                    self.left_hand_pos, left_hand_goal, self.arm_frequency, self.arm_time)
                self.right_traj = mjtg(
                    self.left_hand_pos, right_hand_goal, self.arm_frequency, self.arm_time)
                print("Left traj: ", self.left_traj,
                      ", right traj: ", self.right_traj)
                self.arm_idx += 1
                return self.left_traj[:, 0], self.right_traj[:, 0]
            else:
                self.arm_idx = self.arm_idx + \
                    1 if self.arm_idx < self.left_traj.shape[1] - \
                    1 else self.arm_idx
                print(" Arm idx: %d" % self.arm_idx)
                print("Left traj: ", self.left_traj[:, self.arm_idx],
                      ", right traj: ", self.right_traj[:, self.arm_idx])

                return self.left_traj[:, self.arm_idx], self.right_traj[:, self.arm_idx]
        else:
            return left_hand_goal, right_hand_goal

    def get_q_upper_from_action(self):
        if self.arm_action == 2:
            time_start = 1.0
            left_offset = np.array(
                [0, 0.3, 0.0]) if self.time < time_start else np.array([0, 0.0, 0.0])
            right_offset = np.array(
                [0, -0.3, 0.0]) if self.time < time_start else np.array([0, 0.0, 0.0])

            left_target, right_target = self.getHandSetPoint(
                self.left_reach_goal_box+left_offset,
                self.right_reach_goal_box+right_offset)

            self._left_target = copy.copy(left_target)
            self._right_target = copy.copy(right_target)
            q_upper = list(self.getJointPositionForHands(lhand_pos=left_target,
                                                         rhand_pos=right_target))[:8]

            return q_upper
        elif self.arm_action == 1:
            left_target, right_target = self.getHandSetPoint(
                [0, 0, 0], self.right_reach_goal_door)
            self._left_target = copy.copy(left_target)
            self._right_target = copy.copy(right_target)
            right_hand = list(self.getJointPositionForHands(
                rhand_pos=right_target))[:4]
            q_upper = list(right_hand) + [0.30, -1.35, 0.0, -0.785]
            return q_upper
        else:
            assert self.arm_action == 0
            if self.useMinimumJerk:
                joint_info = [
                    "rightShoulderPitch",
                    "rightShoulderRoll",
                    "rightShoulderYaw",
                    "rightElbowPitch",
                    "leftShoulderPitch",
                    "leftShoulderRoll",
                    "leftShoulderYaw",
                    "leftElbowPitch",
                ]

                current_arm_joints = [self.joint_states[jointName][0]
                                      for jointName in joint_info]
                target = [0.30, 1.35, 0.0, 0.785, 0.30, -1.35, 0.0, -0.785]
                if self.remaining_time > 1/self.PD_freq:
                    retVal = mjtg(current_arm_joints, target,
                                  self.arm_frequency, self.remaining_time)
                    self._right_target = copy.copy(retVal[:4])
                    self._left_target = copy.copy(retVal[-4:])
                    return list(retVal[:, 0])

                else:
                    self._right_target = copy.copy(target[:4])
                    self._left_target = copy.copy(target[-4:])
                    return target
            else:
                if self.arm_counter < 25:
                    retVal = [-0.38, 1.45,  0.15,  1.,
                              -0.38, -1.45, 0.15, -1.]
                    self.arm_counter += 1
                else:
                    retVal = [0.30, 1.35, 0.0, 0.785, 0.30, -1.35, 0.0, -0.785]
                return retVal

    def set_action(self, action):
        """ potential actions: """
        """
        goal at (0,0), (2,0) (4,0) for door
        goal at (0,0), (2,0) (0,0) for clamp

        arm pos: retreat / retract

        lower body: freeze / walk
        => 3*2*2 = 12 actions
        """

        if self.incremental_control_high:
            self.pelvis_goal[0] += copy.copy(
                self.action_combination[action][0][0])
            self.pelvis_goal[1] += copy.copy(
                self.action_combination[action][0][1])
        else:
            self.pelvis_goal[0] = copy.copy(
                self.action_combination[action][0][0])
            self.pelvis_goal[1] = copy.copy(
                self.action_combination[action][0][1])
        if self.arm_action != self.action_combination[action][1]:
            self.remaining_time = self.arm_time
        self.arm_action = copy.copy(self.action_combination[action][1])

        self.stop_gait = copy.copy(self.action_combination[action][2])

    def step(self, action):

        self.obs_snapshot = copy.copy(self.obs)

        counter = 0

        self.high_level_action = copy.copy(action)

        for _ in range(self.high_level_repeat):
            counter += 1
            if self.control_mode == "whole_body":
                self.is_at_goal = np.linalg.norm(
                    np.array(self.end_goal) - np.array([0.6, 0, 0]) - np.array(self.base_pos)) < 0.25 or self.is_at_goal  # is at goal until the goal moved away (time to grab the box)
                self.set_action(copy.copy(self.high_level_action))
                q_upper = self.get_q_upper_from_action()
                self.q_upper = copy.copy(q_upper)

                # for q learning, if robot is after door, then the door has to be open. If spawn behind door, manually open the door
                if (self.middle_goal[0] - 0.2) < self.base_pos[0] or self.is_at_middle_goal:
                    self.is_at_middle_goal = True

                if self.leftHand_isInContact and self.rightHand_isInContact:
                    self.box_in_hand = True
                else:
                    self.box_in_hand = False

            else:
                self.is_at_goal = np.linalg.norm(
                    np.array(self.pelvis_goal) - np.array(self.base_pos)) < 0.15 or self.is_at_goal  # is at goal until the goal moved away (time to grab the box)

                self.stop_gait = self.is_at_goal
                if not self.incremental_control_high:
                    if self.is_at_goal and self.door_is_open:
                        print("=====================")
                        print("Moving goal")
                        self.pelvis_goal = np.random.uniform(
                            self.pelvis_range_low, self.pelvis_range_high)
                        self.is_at_goal = False

                q_upper = list(self.getJointPositionForHands()) if (
                    not self.door_is_open and self.spawn_objects) else list(self.q_nom_list[:8])

            if self.base_pos[0] < 4.:
                lower_policy = self.lower_policy_side
            else:
                lower_policy = self.lower_policy_straight
                self.changePDgains()

            with lower_policy.set_deterministic(True):
                q_lower = np.array(
                    lower_policy.actions_np(self.walk_obs[None])[0])

                low, high = np.array(
                    self.jointLowerLimit[-8:]), np.array(self.jointUpperLimit[-8:])
                q_lower = low + (q_lower + 1.0) * (high - low) / 2.0
                q_lower = list(q_lower)

            if self.control_mode == "upper":
                q_upper = np.array(q_upper) + np.array(action)
                q_upper = list(q_upper)
            elif self.control_mode == "lower":
                q_lower = np.array(q_lower) + np.array(action)
                q_lower = list(q_lower)
            elif self.control_mode == "whole_body":
                pass  # this was handled at beginning of step(self, action)
            else:
                raise ValueError("Control mode %s does not exist" %
                                 self.control_mode)

            action = q_upper + q_lower

            self.q_lower = copy.copy(q_lower)
            if self.incremental_control:
                action = np.clip(action, self.action_space.low,
                                 self.action_space.high)
                self.action = self.getJointPositionForHands() + copy.copy(action)
            else:
                self.action = copy.copy(action)

            self.action = np.clip(
                self.action, self.jointLowerLimitAbsolute, self.jointUpperLimitAbsolute)

            self.stabilised = self.time >= self.time_to_stabilise
            # if not +1, then doubled time to stabilise because of > check
            ticks = self._actionRepeat if self.stabilised else int(
                np.ceil(self.time_to_stabilise*self.Physics_freq))+1

            assert len(self.action) == len(self.q_nom_list)
            if not self.stabilised:
                self.action = self.q_nom_list

            raw_action = copy.copy(self.action)

            if self.interpolate_action:
                self.joint_interpolater.cubic_interpolation_setup(
                    self.prev_action, .0, raw_action, .0, self._dt_filter)

            for _ in range(ticks):
                self.action = self.getFilteredAction(raw_action)
                self.set_pos(self.action)
                """Push force"""
                if self.exertForce:
                    for dist_time in self.disturbance_time:
                        if (self.time >= dist_time[0]) and (self.time <= dist_time[1]):
                            t = self.time-dist_time[0]
                            a = -4/(self.force_duration**2)
                            b = 4/self.force_duration

                            def force_x(t): return (a*t**2+b*t)*dist_time[2]
                            def force_y(t): return (a*t**2+b*t)*dist_time[3]

                            self._p.applyExternalForce(
                                self.robot, -1,
                                forceObj=[force_x(t), force_y(t), 0],
                                posObj=[0, 0, 0],
                                flags=self._p.LINK_FRAME)
                            print("Pushing with %.2f at %.2f" %
                                  (force_x(t), self.time))
                """Doing sim step"""
                self._p.stepSimulation()

                self.time += 1/self.Physics_freq

                if self.save_trajectories:

                    for name in self.controlled_joints:
                        _id = self.jointIdx[name]

                        self.joint_states.update(
                            {self.jointNameIdx[_id]: self._p.getJointState(self.robot, _id)})
                    self.time_array.append(self.time)
                    joint_info = [
                        "rightShoulderPitch",
                        "rightShoulderRoll",
                        "rightShoulderYaw",
                        "rightElbowPitch",
                        "leftShoulderPitch",
                        "leftShoulderRoll",
                        "leftShoulderYaw",
                        "leftElbowPitch",
                        "rightHipRoll",
                        "rightHipPitch",
                        "rightKneePitch",
                        "rightAnklePitch",
                        "leftHipRoll",
                        "leftHipPitch",
                        "leftKneePitch",
                        "leftAnklePitch",
                    ]

                    joint_position = [self.joint_states[jointName][0]
                                      for jointName in joint_info]
                    self.box_pos, self.box_quat = self._p.getBasePositionAndOrientation(
                        self.box)

                    self.box_traj.append(self.box_pos)
                    self.q_real_traj.append(copy.copy(joint_position))
                    self.action_traj.append(
                        copy.copy(list(self.q_upper))+copy.copy(list(self.q_lower)))
                    queried_links = [
                        "head_imu_joint",
                        "leftElbowPitch",
                        "rightElbowPitch",
                        "left_hand",
                        "right_hand",
                        "torsoYaw",
                        "leftKneePitch",
                        "rightKneePitch",
                        "leftAnklePitch",
                        "rightAnklePitch",
                    ]
                    queried_indices = []
                    for link in queried_links:
                        queried_indices.append(self.jointIdx[link])
                    self.linkStates = self._p.getLinkStates(
                        self.robot, queried_indices, computeLinkVelocity=True)
                    self.left_foot_pos = self.linkStates[8][0]
                    self.right_foot_pos = self.linkStates[9][0]

                    self.left_foot_orientation = self._p.getEulerFromQuaternion(
                        self.linkStates[8][1])
                    self.right_foot_orientation = self._p.getEulerFromQuaternion(
                        self.linkStates[9][1])
                    self.base_pos, self.base_quat = self._p.getBasePositionAndOrientation(
                        self.robot)
                    self.eef_pose_traj.append(list(np.array(self.base_pos) - np.array(self.left_foot_pos))+[self.left_foot_orientation[1]]
                                              + list(np.array(self.base_pos) - np.array(self.right_foot_pos))+[self.right_foot_orientation[1]])
                    self.pelvis_goal_traj.append(
                        copy.copy(self.pelvis_goal[:2]))

                    self.left_hand_pos = self.linkStates[3][0]
                    self.right_hand_pos = self.linkStates[4][0]
                    self.hands_pose_traj.append(
                        list(self.left_hand_pos) + list(self.right_hand_pos))
                    self.hands_pose_target.append(
                        list(self._left_target)+list(self._right_target))
                    self.com_traj.append(self.calCOMPos())

                if self.links_to_read is not None:
                    frame_list = []
                    for link_to_read in self.links_to_read:
                        frame_list.append(self.getFrame(link_to_read))
                    self.frames.append(
                        dict(zip(self.links_to_read, frame_list)))
                    frame_list = []
                    obstacle_links_to_read = []
                    """Obstacle ids need to be sorted here"""
                    if self.obstacle_ids is not None:
                        for obstacle_idx, obstacle_id in enumerate(self.obstacle_ids):
                            # door
                            if self.door == obstacle_id:
                                assert obstacle_idx == 0
                                obstacle_links_to_read.append("obstacle0")
                                obstacle_links_to_read.append("obstacle1")

                                frames = self.getObstacleFrame(obstacle_id)
                                frame_list.append(frames[0])
                                frame_list.append(frames[1])
                            else:
                                # table and box
                                obstacle_links_to_read.append(
                                    "obstacle%d" % (obstacle_idx+1))
                                frame_list.append(
                                    self.getObstacleFrame(obstacle_id))

                        self.obstacle_frames.append(
                            dict(zip(obstacle_links_to_read, frame_list)))

                        self.obstacle_links_to_read = obstacle_links_to_read

            obs = self.get_observation()

        reward, reward_info = copy.copy(self.getReward())

        done = copy.copy(self.checkFall())

        self.prev_action = copy.copy(self.action)
        return copy.copy(obs), reward, done, reward_info

    def getFrame(self, linkName):
        if linkName != 'pelvis':
            state = self._p.getLinkState(self.robot, self.linkIdx[linkName])
            frame = state[4:6]
        else:
            frame = self._p.getBasePositionAndOrientation(self.robot)
        return frame

    def getObstacleFrame(self, obstacle_id):
        positionA, orientationA = self._p.getBasePositionAndOrientation(
            obstacle_id)

        if obstacle_id == self.door:
            state = self._p.getLinkState(obstacle_id, 1)
            positionB = state[0]
            orientationB = state[1]
            frame = [positionA, orientationA]
            return frame, [positionB, orientationB]
        else:
            frame = [positionA, orientationA]
            return frame

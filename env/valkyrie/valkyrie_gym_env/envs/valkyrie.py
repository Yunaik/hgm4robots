from pybullet_utils.bullet_client import BulletClient
import inspect
import scipy
import random
import gym
import copy
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as pybullet
import math
from .filter import FilterClass
from .motion import Motion
from .interpolation import JointTrajectoryInterpolate
import pybullet_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
__package__ = "valkyrie_gym_env"
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


class Valkyrie(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 renders=False,
                 Kp_scale=None,
                 Kd_scale=None,
                 planeId=None,
                 fixed_base=False,
                 time_step=0.005,
                 frame_skip=8,
                 urdf_version=0,
                 start_index=11,  # base vel, angular vel, gravity vector, and goal for pelvis
                 obs_use_foot_force=False,
                 obs_use_foot_pose=False,
                 obs_use_foot_past_action=False,
                 obs_use_pos=False,
                 obs_use_yaw=False,
                 margin_in_degree=0.1,  # if > 90. then read from urdf
                 useFullDOF=True,
                 regularise_action=True,
                 time_to_stabilise=1.,
                 incremental_control=False,
                 controlled_joints=None,
                 not_reaching=True,
                 pelvis_goal=None,
                 reach_short_distance=False,
                 goal_type=None,
                 visualise_goal=False,
                 filter_observation=False,
                 filter_action=False,
                 action_bandwidth=4,
                 interpolate_action=False,
                 terminate_if_not_double_support=False,
                 terminate_if_flight_phase=False,
                 terminate_if_pelvis_out_of_range=False,
                 replayRefMotion=False,
                 imitate_motion=False,
                 imitateWalking=True,
                 imitatePitchOnly=True,
                 target_velocity=[0.5, 0, 0],
                 require_full_contact_foot=False,
                 lock_upper_body=False,
                 weight_dic=None,
                 imit_weights={"imitation": 0.5, "goal": 0.5},
                 joint_weights=None,
                 joint_imit_tolerance=None,
                 print_reward_details=False,
                 save_trajectories=False,
                 allow_faster_vel=False,
                 imitate_ankle_pitch=True,
                 imitate_contact_type=1,
                 dsp_duration=0.2,
                 calculate_PD_from_torque=False,
                 imitate_only_vertical_eef=True,
                 gravity_compensation=False,
                 lateral_friction=1.5,
                 learn_stand=False,
                 exertForce=False,
                 force_duration=0.1,
                 maxImpulse=100,
                 probability_of_push=0.8,
                 goal_as_vel=True,
                 goal_y_range=0.0,
                 tighter_tolerance_upon_reaching_goal=False,
                 q_nom=None,
                 random_joint_init=False,
                 final_goal_type=None,  # "right"
                 whole_body_mode=False,
                 base_pos_spawn_offset=[0, 0, 0],
                 random_spawn=False,
                 debug=False,
                 applySpawnOffset=True,
                 fixed_disturbance=False,
                 replace_foot=0  # 1 is left, 2 is right
                 ):
        """Physics stuff"""
        self.replace_foot = replace_foot
        self.fixed_disturbance = fixed_disturbance
        self.applySpawnOffset = applySpawnOffset
        self.debug = debug
        self.random_spawn = random_spawn
        self.base_pos_spawn_offset = base_pos_spawn_offset
        self.regularise_action = regularise_action
        self.useFullDOF = useFullDOF
        self.timestep = time_step
        self.frame_skip = frame_skip
        self.robot_loaded = False
        self.fixed_base = fixed_base
        self.planeID = planeId

        self.robot = -1
        self.PD_freq = 1/(self.timestep*self.frame_skip)
        self.Physics_freq = 1/self.timestep
        self._actionRepeat = int(self.Physics_freq/self.PD_freq)
        self._dt_physics = (1. / self.Physics_freq)
        self._dt_PD = (1. / self.PD_freq)
        self._dt = self._dt_physics  # PD control loop timestep
        self._dt_filter = self._dt_PD  # filter time step
        self.g = 9.81

        """Learning stuff"""
        self.whole_body_mode = whole_body_mode
        self.x_offset = 0.0 if self.whole_body_mode else 0.0

        self.final_goal = [4.5+self.x_offset, -1.0, 1.175]
        table_offset = 0.75  # this is required because the final goal is where the robot is
        self.table1_pos = [0.5+table_offset+self.x_offset, 0.5, 0.0]
        self.table2_pos = [2.5+table_offset, -0.5, 0.0]

        self.box_pos_nom = [0.5+0.3+self.x_offset, 0.0, 1.0]
        self.box_pos_nom2 = [2.5+0.3+self.x_offset, -1.0, 1.0]

        self.door_pos = [5., -1.5/2-0.025-2., 0]
        self.box_pos = copy.copy(self.box_pos_nom)

        self.start_count = 0
        self.stop_gait = False
        self.box_grasped = False
        self.door_is_open = False
        self.random_joint_init = random_joint_init
        self.tighter_tolerance_upon_reaching_goal = tighter_tolerance_upon_reaching_goal
        self.goal_y_range = goal_y_range
        self.vel_target = target_velocity
        self.goal_as_vel = goal_as_vel
        self.probability_of_push = probability_of_push
        self.exertForce = exertForce
        self.maxImpulse = maxImpulse
        self.learn_stand = learn_stand
        self.lateral_friction = lateral_friction

        self.imitate_only_vertical_eef = imitate_only_vertical_eef
        self.filter_observation = filter_observation
        self.interpolate_action = interpolate_action
        self.calculate_PD_from_torque = calculate_PD_from_torque
        self.urdf_version = urdf_version
        self.save_trajectories = save_trajectories
        if self.save_trajectories:
            self.time_array = []
            self.q_imit_traj = []
            self.q_real_traj = []
            self.action_traj = []
            self.eef_imit_traj = []
            self.eef_pose_traj = []
            self.hands_pose_traj = []
            self.hands_pose_target = []
            self.com_traj = []
            self.vel_traj = []
            self.foot_force = []
            self.box_traj = []
            self.pd_target = []
            self.pd_value = []
            if goal_type is not None:
                self.pelvis_pos_traj = []
                self.pelvis_goal_traj = []
                self.vel_goal_traj = []
                self.grav_traj = []
                self.gravity_goal_traj = []
            self.eef_contact_traj = []
        self.dsp_duration = dsp_duration
        self.imitate_contact_type = imitate_contact_type
        self.imitate_ankle_pitch = imitate_ankle_pitch
        self.allow_faster_vel = allow_faster_vel
        self.imitatePitchOnly = imitatePitchOnly
        self.print_reward_details = print_reward_details

        """Imit reward stuff"""
        self.joint_imit_tolerance = joint_imit_tolerance

        """Goal reward weights"""
        self.final_goal_type = final_goal_type
        assert self.final_goal_type == "right" or self.final_goal_type is None
        self.left_reach_goal_height = 0.97
        self.right_reach_goal_height = 0.97
        if not_reaching:
            self.weight_x_pos_reward = weight_dic["weight_x_pos_reward"] if weight_dic is not None else 2.0
            self.weight_y_pos_reward = weight_dic["weight_y_pos_reward"] if weight_dic is not None else 2.0
            self.weight_z_pos_reward = weight_dic["weight_z_pos_reward"] if weight_dic is not None else 2.0
            self.weight_x_vel_reward = weight_dic["weight_x_vel_reward"] if weight_dic is not None else 6.0
            self.weight_y_vel_reward = weight_dic["weight_y_vel_reward"] if weight_dic is not None else 2.0
            self.weight_z_vel_reward = weight_dic["weight_z_vel_reward"] if weight_dic is not None else 2.0
            self.weight_torso_pitch_reward = weight_dic[
                "weight_torso_pitch_reward"] if weight_dic is not None else 0.5
            self.weight_pelvis_pitch_reward = weight_dic[
                "weight_pelvis_pitch_reward"] if weight_dic is not None else 0.5
            self.weight_left_foot_force_reward = weight_dic[
                "weight_left_foot_force_reward"] if weight_dic is not None else 1.0
            self.weight_right_foot_force_reward = weight_dic[
                "weight_right_foot_force_reward"] if weight_dic is not None else 1.0
            self.weight_joint_vel_reward = weight_dic["weight_joint_vel_reward"] if weight_dic is not None else 1.0
            self.weight_joint_torque_reward = weight_dic[
                "weight_joint_torque_reward"] if weight_dic is not None else 1.0
            self.weight_foot_clearance_reward = weight_dic[
                "weight_foot_clearance_reward"] if weight_dic is not None else 1.0
            self.weight_foot_slippage_reward = weight_dic[
                "weight_foot_slippage_reward"] if weight_dic is not None else 1.0
            self.weight_foot_pitch_reward = weight_dic[
                "weight_foot_pitch_reward"] if weight_dic is not None else 1.0
            self.weight_foot_contact_reward = weight_dic[
                "weight_foot_contact_reward"] if weight_dic is not None else 2.0
            try:
                self.weight_contact_penalty = weight_dic[
                    "weight_contact_penalty"] if weight_dic is not None else 0.0
            except:
                self.weight_contact_penalty = 0.0
            self.weight_gravity_reward = weight_dic["weight_gravity_reward"] if weight_dic is not None else 1.0
            self.weight_imit_joint_pos_reward = weight_dic[
                "imit_joint_pos_reward"] if weight_dic is not None else 0.5
            self.weight_imit_eef_contact_reward = weight_dic[
                "imit_eef_contact_reward"] if weight_dic is not None else 0.2
            self.weight_imit_eef_pos_reward = weight_dic[
                "imit_eef_pos_reward"] if weight_dic is not None else 0.2
            self.weight_imit_eef_orientation_reward = weight_dic[
                "imit_eef_orientation_reward"] if weight_dic is not None else 0.1
        else:
            self.reach_weight_dic = weight_dic

        self.imit_weights = imit_weights
        self.lock_upper_body = lock_upper_body
        self.require_full_contact_foot = require_full_contact_foot
        self.target_velocity = target_velocity
        self.imitate_motion = imitate_motion
        self.imitateWalking = imitateWalking

        if self.imitate_motion:
            time_to_stabilise = 0.
        self.replayRefMotion = replayRefMotion
        if self.replayRefMotion:
            print("==========================")
            print("CAREFUL REPLAY MOTION for debugging is active")
        self.action_bandwidth = action_bandwidth
        self.filter_action = filter_action

        self.terminate_if_pelvis_out_of_range = terminate_if_pelvis_out_of_range
        self.terminate_if_flight_phase = terminate_if_flight_phase
        self.terminate_if_not_double_support = terminate_if_not_double_support
        self.visualise_goal = visualise_goal
        self.not_reaching = not_reaching
        """ Modifying goal pos for pelvis"""
        # if not_reaching:
        self.goal_update_time = 1/self.PD_freq  # 0.3
        self.time_last_move_goal = -1

        self.goal_type = goal_type
        self.pelvis_goal = pelvis_goal

        if self.imitate_motion:
            self.pelvis_range_low = np.array(
                [3.0/(1+reach_short_distance), -self.goal_y_range, 1.175])
            self.pelvis_range_high = np.array(
                [5.0/(1+reach_short_distance),  self.goal_y_range, 1.175])
        else:
            self.pelvis_range_low = np.array(
                [-0.07, -0.1*useFullDOF, 0.97])
            self.pelvis_range_high = np.array(
                [0.1,   0.1*useFullDOF, 1.12])

        self.goal_increment_max = 0.04  # 1cm every second

        if self.goal_type == "fixed":
            self.pelvis_goal = self.pelvis_goal if self.pelvis_goal is not None else [
                0.0, 0.0, 1.175]
        elif self.goal_type == "random_fixed":
            self.pelvis_goal = np.random.uniform(
                self.pelvis_range_low, self.pelvis_range_high)
        elif self.goal_type == "moving_goal":
            self.pelvis_goal = np.random.uniform(
                [0.5, -self.goal_y_range/2, 1.175], [0.5, self.goal_y_range/2, 1.175])
        elif self.goal_type is None:
            self.pelvis_goal = [0.0, 0.0, 1.175]  # not used
        elif self.goal_type == "fixed_behind":
            self.pelvis_goal = [-2.0, 0.0, 1.175]
        else:
            print("Reach type %s not defined" % self.goal_type)
            assert 3 == 4, "Reach type not defined"

        if self.learn_stand:
            self.pelvis_goal[2] = 1.06
        """Other stuff"""
        self.gravity_compensation = gravity_compensation
        self.max_torque_scale = 1/5. if self.gravity_compensation else 1.
        self.incremental_control = incremental_control
        self.stabilised = False
        self.time = 0
        self.time_to_stabilise = time_to_stabilise

        self.Kp_scale = dict([
            ("rightShoulderPitch",  1),
            ("rightShoulderRoll",   1),
            ("rightShoulderYaw",    1),
            ("rightElbowPitch",     1),
            ("leftShoulderPitch",   1),
            ("leftShoulderRoll",    1),
            ("leftShoulderYaw",     1),
            ("leftElbowPitch",      1),
            ("torsoYaw",            1),
            ("torsoPitch",          1),
            ("torsoRoll",           1),
            ("leftHipYaw",          1),
            ("leftHipRoll",         1),
            ("leftHipPitch",        1),
            ("leftKneePitch",       2),
            ("leftAnklePitch",      2),
            ("leftAnkleRoll",       1),
            ("rightHipYaw",         1),
            ("rightHipRoll",        1),
            ("rightHipPitch",       1),
            ("rightKneePitch",      2),
            ("rightAnklePitch",     2),
            ("rightAnkleRoll",      1),
        ]) if Kp_scale is None else Kp_scale
        self.Kd_scale = dict([
            ("rightShoulderPitch",  1),
            ("rightShoulderRoll",   1),
            ("rightShoulderYaw",    1),
            ("rightElbowPitch",     1),
            ("leftShoulderPitch",   1),
            ("leftShoulderRoll",    1),
            ("leftShoulderYaw",     1),
            ("leftElbowPitch",      1),
            ("torsoYaw",            1),
            ("torsoPitch",          1),
            ("torsoRoll",           1),
            ("leftHipYaw",          1),
            ("leftHipRoll",         1),
            ("leftHipPitch",        1),
            ("leftKneePitch",       1),
            ("leftAnklePitch",      1),
            ("leftAnkleRoll",       1),
            ("rightHipYaw",         1),
            ("rightHipRoll",        1),
            ("rightHipPitch",       1),
            ("rightKneePitch",      1),
            ("rightAnklePitch",     1),
            ("rightAnkleRoll",      1),
        ]) if Kd_scale is None else Kd_scale

        self.jointIdx = {}
        self.jointNameIdx = {}

        self.linkIdx = {}
        self.linkNameIdx = {}

        self.jointLowerLimit = []
        self.jointUpperLimit = []
        self.total_mass = 0.0

        self._envStepCounter = 0
        self._renders = renders
        self._p = BulletClient(connection_mode=pybullet.DIRECT) if not renders else BulletClient(
            connection_mode=pybullet.GUI)
        if self.useFullDOF:
            if self.lock_upper_body:
                self.controlled_joints = [
                    "rightHipRoll",
                    "rightHipPitch",
                    "rightKneePitch",
                    "rightAnklePitch",
                    "leftHipRoll",
                    "leftHipPitch",
                    "leftKneePitch",
                    "leftAnklePitch",
                    # "leftAnkleRoll",
                ] if controlled_joints is None else controlled_joints
            else:
                self.controlled_joints = [
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
                    # "leftAnkleRoll",
                ] if controlled_joints is None else controlled_joints

        else:
            self.controlled_joints = [
                "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch",
                "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch", ] if controlled_joints is None else controlled_joints

        self.nu = len(self.controlled_joints)

        """Imitation stuff"""
        if self.imitate_motion:
            dsr_gait_freq = 0.6 if imitateWalking else None

            self.imitating_joints = [
                "rightHipRoll",
                "rightHipPitch",
                "rightKneePitch",
                "rightAnklePitch",
                "leftHipRoll",
                "leftHipPitch",
                "leftKneePitch",
                "leftAnklePitch",
            ] if self.imitatePitchOnly else self.controlled_joints

            self.human_motion = Motion(controlled_joints=self.imitating_joints,
                                       dsr_data_freq=self.PD_freq, dsr_gait_freq=dsr_gait_freq, stand_config=None)
            self.joint_weights = {
                "rightHipRoll": 1,
                "rightHipPitch": 4,
                "rightKneePitch": 2,
                "rightAnklePitch": 1*self.imitate_ankle_pitch,
                "leftHipRoll": 1,
                "leftHipPitch": 4,
                "leftKneePitch": 2,
                "leftAnklePitch": 1*self.imitate_ankle_pitch,
            } if joint_weights is None else joint_weights

        scaling = 2 if self.learn_stand else 1.
        self.joint_states = {}
        self.u_max = dict([("torsoYaw", 190),
                           ("torsoPitch", 150),
                           ("torsoRoll", 150),
                           ("rightShoulderPitch", 190),
                           ("rightShoulderRoll", 190),
                           ("rightShoulderYaw", 65),
                           ("rightElbowPitch", 65),
                           ("rightForearmYaw", 26),
                           ("rightWristRoll", 14),
                           ("rightWristPitch", 14),
                           ("leftShoulderPitch", 190),
                           ("leftShoulderRoll", 190),
                           ("leftShoulderYaw", 65),
                           ("leftElbowPitch", 65),
                           ("leftForearmYaw", 26),
                           ("leftWristRoll", 14),
                           ("leftWristPitch", 14),
                           ("rightHipYaw", 190),
                           ("rightHipRoll", 350),
                           ("rightHipPitch", 350),
                           ("rightKneePitch", 350),
                           ("rightAnklePitch", 205/scaling),
                           ("rightAnkleRoll", 205),
                           ("leftHipYaw", 190),
                           ("leftHipRoll", 350),
                           ("leftHipPitch", 350),
                           ("leftKneePitch", 350),
                           ("leftAnklePitch", 205/scaling),
                           ("leftAnkleRoll", 205),
                           ("lowerNeckPitch", 50),
                           ("upperNeckPitch", 50),
                           ("neckYaw", 50)])

        self.v_max = dict([("torsoYaw", 5.89),
                           ("torsoPitch", 9),
                           ("torsoRoll", 9),
                           ("rightShoulderPitch", 5.89),
                           ("rightShoulderRoll", 5.89),
                           ("rightShoulderYaw", 11.5),
                           ("rightElbowPitch", 11.5),
                           ("leftShoulderPitch", 5.89),
                           ("leftShoulderRoll", 5.89),
                           ("leftShoulderYaw", 11.5),
                           ("leftElbowPitch", 11.5),
                           ("rightHipYaw", 5.89),
                           ("rightHipRoll", 7),
                           ("rightHipPitch", 6.11),
                           ("rightKneePitch", 6.11),
                           ("rightAnklePitch", 11),
                           ("rightAnkleRoll", 11),
                           ("leftHipYaw", 5.89),
                           ("leftHipRoll", 7),
                           ("leftHipPitch", 6.11),
                           ("leftKneePitch", 6.11),
                           ("leftAnklePitch", 11),
                           ("leftAnkleRoll", 11),
                           ("lowerNeckPitch", 5),
                           ("upperNeckPitch", 5),
                           ("neckYaw", 5)])

        if self.useFullDOF:
            self.q_nom = dict([
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
                ("rightShoulderRoll", 1.25),
                ("rightShoulderYaw", 0.0),
                ("rightElbowPitch", 0.785398163397),
                ("leftShoulderPitch", 0.300196631343),
                ("leftShoulderRoll", -1.25),
                ("leftShoulderYaw", 0.0),
                ("leftElbowPitch", -0.785398163397),
            ]) if q_nom is None else q_nom

        else:
            self.q_nom = dict([
                ("rightHipPitch",    -0.45*1.),
                ("rightKneePitch",   0.944*1.),
                ("rightAnklePitch",  -0.527*1.),
                ("leftHipPitch",    -0.45*1.),
                ("leftKneePitch",    0.944*1.),
                ("leftAnklePitch",  -0.527*1.),
            ]) if q_nom is None else q_nom

        margin = margin_in_degree*3.14/180
        if self.debug:
            print(
                "Margin: %.1f[deg] (if larger than 90 degree, using joint limits from urdf)" % margin_in_degree)
        self.margin = margin
        if self.useFullDOF:
            self.joint_limits_low = {
                "rightShoulderPitch": (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["rightShoulderPitch"],
                "rightShoulderRoll": (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["rightShoulderRoll"],
                "rightShoulderYaw":  (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["rightShoulderYaw"],
                "rightElbowPitch":   (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["rightElbowPitch"],
                "leftShoulderPitch": (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["leftShoulderPitch"],
                "leftShoulderRoll":  (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["leftShoulderRoll"],
                "leftShoulderYaw":   (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["leftShoulderYaw"],
                "leftElbowPitch":    (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["leftElbowPitch"],
                "torsoYaw":          (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["torsoYaw"],
                "torsoPitch": -margin+self.q_nom["torsoPitch"],
                "torsoRoll":         (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["torsoRoll"],
                "rightHipYaw":       (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["rightHipYaw"],
                "rightHipRoll": -margin+self.q_nom["rightHipRoll"],
                "rightHipPitch": -margin+self.q_nom["rightHipPitch"],
                "rightKneePitch": -margin+self.q_nom["rightKneePitch"],
                "rightAnklePitch": -margin+self.q_nom["rightAnklePitch"],
                "rightAnkleRoll": -margin+self.q_nom["rightAnkleRoll"],
                "leftHipYaw":        (not self.lock_upper_body)*(-margin)-1e-3+self.q_nom["leftHipYaw"],
                "leftHipRoll": -margin+self.q_nom["leftHipRoll"],
                "leftHipPitch": -margin+self.q_nom["leftHipPitch"],
                "leftKneePitch": -margin+self.q_nom["leftKneePitch"],
                "leftAnklePitch": -margin+self.q_nom["leftAnklePitch"],
                "leftAnkleRoll": -margin+self.q_nom["leftAnkleRoll"]}

            self.joint_limits_high = {
                "rightShoulderPitch": (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["rightShoulderPitch"],
                "rightShoulderRoll": (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["rightShoulderRoll"],
                "rightShoulderYaw":  (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["rightShoulderYaw"],
                "rightElbowPitch":   (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["rightElbowPitch"],
                "leftShoulderPitch": (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["leftShoulderPitch"],
                "leftShoulderRoll":  (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["leftShoulderRoll"],
                "leftShoulderYaw":   (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["leftShoulderYaw"],
                "leftElbowPitch":    (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["leftElbowPitch"],
                "torsoYaw":          (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["torsoYaw"],
                "torsoPitch": +margin+self.q_nom["torsoPitch"],
                "torsoRoll":         (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["torsoRoll"],
                "rightHipYaw":     (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["rightHipYaw"],
                "rightHipRoll": +margin+self.q_nom["rightHipRoll"],
                "rightHipPitch": +margin+self.q_nom["rightHipPitch"],
                "rightKneePitch": +margin+self.q_nom["rightKneePitch"],
                "rightAnklePitch": +margin+self.q_nom["rightAnklePitch"],
                "rightAnkleRoll": +margin+self.q_nom["rightAnkleRoll"],
                "leftHipYaw":      (not self.lock_upper_body)*(+margin)+1e-3+self.q_nom["leftHipYaw"],
                "leftHipRoll": +margin+self.q_nom["leftHipRoll"],
                "leftHipPitch": +margin+self.q_nom["leftHipPitch"],
                "leftKneePitch": +margin+self.q_nom["leftKneePitch"],
                "leftAnklePitch": +margin+self.q_nom["leftAnklePitch"],
                "leftAnkleRoll": +margin+self.q_nom["leftAnkleRoll"]}

        else:
            self.joint_limits_low = {
                "rightHipPitch": -margin+self.q_nom["rightHipPitch"],
                "rightKneePitch": -margin+self.q_nom["rightKneePitch"],
                "rightAnklePitch": -margin+self.q_nom["rightAnklePitch"],
                "leftHipPitch": -margin+self.q_nom["leftHipPitch"],
                "leftKneePitch": -margin+self.q_nom["leftKneePitch"],
                "leftAnklePitch": -margin+self.q_nom["leftAnklePitch"],
            }
            self.joint_limits_high = {
                "rightHipPitch": +margin+self.q_nom["rightHipPitch"],
                "rightKneePitch": +margin+self.q_nom["rightKneePitch"],
                "rightAnklePitch": +margin+self.q_nom["rightAnklePitch"],
                "leftHipPitch": +margin+self.q_nom["leftHipPitch"],
                "leftKneePitch": +margin+self.q_nom["leftKneePitch"],
                "leftAnklePitch": +margin+self.q_nom["leftAnklePitch"],
            }

        if self.imitate_motion:
            self.joint_limits_low, self.joint_limits_high = self.human_motion.getJointRange(
                self.margin)
            self.joint_imit_tolerance_dict = {}
            for jointName in self.imitating_joints:
                if self.joint_imit_tolerance is None:
                    self.joint_imit_tolerance_dict.update({jointName: abs(
                        self.joint_limits_high[jointName]-self.joint_limits_low[jointName])*180/3.14})
                else:
                    self.joint_imit_tolerance_dict.update(
                        {jointName: self.joint_imit_tolerance[jointName]})
        self.linkCOMPos = {}
        self.linkMass = {}
        offset = .1 if self.fixed_base and not_reaching else 0.
        z_height = 1.175 if self.imitate_motion else 1.08

        # 1.175 straight #1.025 bend
        self.base_pos_nom = np.array([0, 0, z_height+offset])
        self.base_orn_nom = np.array([0, 0, 0, 1])  # x,y,z,w
        self.plane_pos_nom = np.array([0., 0., 0.])
        self.plane_orn_nom = np.array([0., 0., 0., 1.])

        self._setupSimulation()
        if self.calculate_PD_from_torque:
            error_for_full_torque = 10*np.pi/180.
            kd_fraction = 1/100.
            self.Kp = {}
            self.Kd = {}
            for name in self.controlled_joints:

                # if gravity compensation is on, then u_max is not full torque, but only a fraction of it (rest comes from gravity compensation)
                val = (self.u_max[name]*self.max_torque_scale) / \
                    error_for_full_torque
                self.Kp.update({name: val*self.Kp_scale[name]})
                self.Kd.update({name: val*kd_fraction*self.Kd_scale[name]})
        else:

            kp_unit = 10
            self.Kp = dict([
                ("torsoYaw", 190 * kp_unit),
                ("torsoPitch", 150 * kp_unit),
                ("torsoRoll", 150 * kp_unit),
                ("rightShoulderPitch", 190 * kp_unit),
                ("rightShoulderRoll", 190 * kp_unit),
                ("rightShoulderYaw", 65 * kp_unit),
                ("rightElbowPitch", 65 * kp_unit),
                ("rightHipYaw", 190 * kp_unit),
                ("rightHipRoll", 350 * kp_unit),
                ("rightHipPitch", 350 * kp_unit),
                ("rightKneePitch", 350 * kp_unit),
                ("rightAnklePitch", 205 * kp_unit),
                ("rightAnkleRoll", 205 * kp_unit),
                ("leftShoulderPitch", 190 * kp_unit),
                ("leftShoulderRoll", 190 * kp_unit),
                ("leftShoulderYaw", 65 * kp_unit),
                ("leftElbowPitch", 65 * kp_unit),
                ("leftHipYaw", 190 * kp_unit),
                ("leftHipRoll", 350 * kp_unit),
                ("leftHipPitch", 350 * kp_unit),
                ("leftKneePitch", 350 * kp_unit),
                ("leftAnklePitch", 205 * kp_unit),
                ("leftAnkleRoll", 205 * kp_unit),
            ])
            kd_unit = 0.2
            self.Kd = dict([
                ("torsoYaw", 190 * kd_unit),
                ("torsoPitch", 150 * kd_unit),
                ("torsoRoll", 150 * kd_unit),
                ("rightShoulderPitch", 190 * kd_unit),
                ("rightShoulderRoll", 190 * kd_unit),
                ("rightShoulderYaw", 65 * kd_unit),
                ("rightElbowPitch", 65 * kd_unit),
                ("rightHipYaw", 190 * kd_unit),
                ("rightHipRoll", 350 * kd_unit),
                ("rightHipPitch", 350 * kd_unit),
                ("rightKneePitch", 350 * kd_unit),
                ("rightAnklePitch", 205 * kd_unit),
                ("rightAnkleRoll", 205 * kd_unit),
                ("leftShoulderPitch", 190 * kd_unit),
                ("leftShoulderRoll", 190 * kd_unit),
                ("leftShoulderYaw", 65 * kd_unit),
                ("leftElbowPitch", 65 * kd_unit),
                ("leftHipYaw", 190 * kd_unit),
                ("leftHipRoll", 350 * kd_unit),
                ("leftHipPitch", 350 * kd_unit),
                ("leftKneePitch", 350 * kd_unit),
                ("leftAnklePitch", 205 * kd_unit),
                ("leftAnkleRoll", 205 * kd_unit),
            ])

        self.getLinkMass()
        """Joint stuff"""
        self.q_nom_list = []
        for jointName in self.controlled_joints:
            self.q_nom_list.append(self.q_nom[jointName])
        self.q_nom_list = np.array(self.q_nom_list)
        if self.imitate_motion:
            self.q_nom_stand = {}
            for name in self.controlled_joints:
                self.q_nom_stand.update({name: self.q_nom[name]})
            self.human_motion.stand_config = self.q_nom_stand

            self.q_nom_list = self.get_first_frame()

        self.action = self.q_nom_list if not self.random_joint_init else self.action_random_init
        self.prev_action = copy.copy(self.action)
        self._actionDim = len(self.controlled_joints)
        if self.debug:
            print("Action dim: ", self._actionDim)
        self.obs_use_foot_force = obs_use_foot_force
        self.obs_use_foot_pose = obs_use_foot_pose
        self.obs_use_foot_past_action = obs_use_foot_past_action
        self.obs_use_pos = obs_use_pos
        self.obs_use_yaw = obs_use_yaw

        self.start_index = (start_index
                            + 2*self.imitate_motion
                            - 2*(self.goal_type is None)  # only x and y, no z
                            + 2*1*self.obs_use_foot_force  # just force z direction
                            + 2*4*self.obs_use_foot_pose  # 3 pos + 1 pitch
                            + 2*self.obs_use_pos  # x and y pos pelvis
                            + 1*self.obs_use_yaw  # yaw of pelvis
                            + self._actionDim*self.obs_use_foot_past_action
                            # +2*(not self.goal_as_vel)
                            )
        # if no goal type, no goal is given to obs
        if self.debug:
            print("Start idx: %d, actionDim: %d" %
                  (self.start_index, self._actionDim))
        observationDim = self.start_index+self._actionDim
        if self.debug:
            print("Obs dim of valkyrie model: ", observationDim)
        observation_high = np.array(
            [np.finfo(np.float32).max] * observationDim)
        self.observation_space = spaces.Box(-observation_high,
                                            observation_high)
        self._observationDim = observationDim
        if self.incremental_control:
            self.action_space = spaces.Box(-np.array(self.joint_increment),
                                           np.array(self.joint_increment))
        else:
            self.action_space = spaces.Box(
                np.array(self.jointLowerLimit), np.array(self.jointUpperLimit))
        """Setting filter up"""

        if self.filter_action:
            filter_order = 1
            self.action_filter_method = FilterClass(self._actionDim)
            # sample period, cutoff frequency, order
            self.action_filter_method.butterworth(
                1./self.Physics_freq, self.action_bandwidth, filter_order)
            for _ in range(1000):
                self.action = self.getFilteredAction(self.q_nom_list)

        if self.filter_observation:
            filter_obs_dim = self._observationDim - (+2*self.imitate_motion
                                                     + 3 *
                                                     (not (self.goal_type is None))
                                                     + self._actionDim*self.obs_use_foot_past_action)
            filter_order = 1

            self.state_filter_method = FilterClass(filter_obs_dim)
            self.state_filter_method.butterworth(
                self._dt_filter, 10, filter_order)

        if self.interpolate_action:
            self.joint_interpolater = JointTrajectoryInterpolate(
                self._actionDim)

        self.vel_list = []
        self.reward_dict_list = {}

        """Force settings"""
        self.force_duration = force_duration

        time1 = np.random.uniform(2.5, 4.5) if np.random.random(
        ) < self.probability_of_push else 10000
        time2 = np.random.uniform(5.0, 7.0) if np.random.random(
        ) < self.probability_of_push else 10000
        time3 = np.random.uniform(7.5, 9.5) if np.random.random(
        ) < self.probability_of_push else 10000

        force_durations = [self.force_duration for i in range(3)]
        force_magnitude_disturbance_x = []
        force_magnitude_disturbance_y = []

        for duration in force_durations:
            total_impulse_magnitude = np.random.uniform(
                -self.maxImpulse, self.maxImpulse)

            y_component = 0

            x_component = total_impulse_magnitude

            x_component /= duration
            y_component /= duration

            force_magnitude_disturbance_x.append(x_component)
            force_magnitude_disturbance_y.append(y_component)

        self.disturbance_time = [
                                [time1, time1+force_durations[0], force_magnitude_disturbance_x[0],
                                    force_magnitude_disturbance_y[0]],
                                [time2, time2+force_durations[1], force_magnitude_disturbance_x[1],
                                    force_magnitude_disturbance_y[1]],
                                [time3, time3+force_durations[2], force_magnitude_disturbance_x[2],
                                    force_magnitude_disturbance_y[2]],
        ]

    def contactWithObject(self):
        contact = False
        if self.final_goal is not None:
            for joint in self.jointIds:
                if (len(self._p.getContactPoints(self.robot, self.table, joint, -1)) > 0):
                    contact = True
                    break
        return contact

    def loadAllObjects(self, loadObject=True, fix_box=False):
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if loadObject:
            self.table_final = self._p.loadURDF(self.dir_path + "/urdf/table.urdf",
                                                basePosition=self.table2_pos,
                                                baseOrientation=[
                                                    0.0, 0.0, 0.707107, 0.707107],
                                                useFixedBase=True)

            self.table = self._p.loadURDF(self.dir_path + "/urdf/table.urdf",
                                          basePosition=self.table1_pos,
                                          baseOrientation=[
                                              0.0, 0.0, 0.707107, 0.707107],
                                          useFixedBase=True)
            box_pos = self.box_pos_nom if self.base_pos_spawn_offset[0] < 2.0 else self.box_pos_nom2

            self.box = self._p.loadURDF(self.dir_path + "/urdf/box_target.urdf",
                                        basePosition=box_pos,
                                        useFixedBase=fix_box,
                                        baseOrientation=[0.0, 0.0, 0.707107, 0.707107])
            self._p.changeDynamics(
                self.robot, self.jointIdx["left_hand"], lateralFriction=2.0)
            self._p.changeDynamics(
                self.robot, self.jointIdx["right_hand"], lateralFriction=2.0)
            self._p.changeDynamics(self.box, -1, lateralFriction=2.0)

            collision = "_collision"
            self.door = self._p.loadURDF(self.dir_path + "/urdf/door%s.urdf" % collision,
                                         basePosition=self.door_pos,
                                         baseOrientation=self._p.getQuaternionFromEuler([0, 0, 3.14/2]))

    def openDoor(self):
        self.door_is_open = True
        self.door_pos_init += (-3*3.14/180) if self.door_pos_init >= -120*3.14 / \
            180 else 0.
        self._p.setJointMotorControl2(
            self.door, 1, self._p.POSITION_CONTROL, targetPosition=self.door_pos_init)

    def closeDoor(self):
        self.door_is_open = False
        self._p.setJointMotorControl2(
            self.door, 1, self._p.POSITION_CONTROL, targetPosition=-(0*3.14/180))

    def getFilteredAction(self, action):
        if self.filter_action:
            self.unfiltered_action = copy.copy(action)
            filtered_values = self.action_filter_method.applyFilter(
                self.unfiltered_action)
            self.filtered_action = filtered_values
            return self.filtered_action
        elif self.interpolate_action:
            self.unfiltered_action = copy.copy(action)
            self.filtered_action = self.joint_interpolater.interpolate(
                self._dt_physics)
            return self.filtered_action
        else:
            self.filtered_action = action
            self.unfiltered_action = action
            return action

    def getLinkMass(self):
        self.total_mass = 0
        info = self._p.getDynamicsInfo(self.robot, -1)  # for base link
        self.linkMass.update({"base": info[0]})
        self.total_mass += info[0]
        for key, value in self.jointIdx.items():
            info = self._p.getDynamicsInfo(self.robot, value)
            self.linkMass.update({key: info[0]})
            self.total_mass += info[0]

    def getLinkCOMPos(self):
        self.linkCOMPos = {}
        base_pos, base_quat = self._p.getBasePositionAndOrientation(self.robot)
        # base position is the COM of the pelvis
        self.linkCOMPos.update({"base": np.array(base_pos)})
        for key, value in self.jointIdx.items():
            info = self._p.getLinkState(
                self.robot, value, computeLinkVelocity=0)
            self.linkCOMPos.update({key: info[0]})

    def calCOMPos(self):
        self.getLinkCOMPos()
        _sum = np.zeros((1, 3))
        for key, value in self.linkMass.items():
            _sum += np.array(self.linkCOMPos[key]) * value
        _sum /= self.total_mass
        return _sum.flatten()

    def step(self, action):
        if self.replayRefMotion:
            ref_joint_pos_dict = self.human_motion.ref_joint_angle_dict(
                request_stand_config=self.learn_stand)
            action = self.q_nom_list
            for n in range(self._actionDim):
                key = self.controlled_joints[n]
                if key in ref_joint_pos_dict.keys():
                    action[n] = ref_joint_pos_dict[key]
            self.incremental_control = False

        if self.incremental_control:
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)
            self.action = self.getJointPositionForHands() + copy.copy(action)
        else:
            self.action = copy.copy(action)

        self.action = np.clip(
            self.action, self.jointLowerLimit, self.jointUpperLimit)

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
                            self.robot, -1, forceObj=[force_x(t), force_y(t), 0], posObj=[0, 0, 0], flags=self._p.LINK_FRAME)
            """Doing sim step"""
            self._p.stepSimulation()
            self.time += 1/self.Physics_freq

            if self.save_trajectories:
                self.time_array.append(self.time)
                self.q_real_traj.append(copy.copy(self.joint_position))
                self.action_traj.append(copy.copy(self.action))
                self.eef_pose_traj.append(list(np.array(self.left_foot_pos))+[self.left_foot_orientation[1]]
                                          + list(np.array(self.right_foot_pos))+[self.right_foot_orientation[1]])
                self.hands_pose_traj.append(
                    list(np.array(self.left_hand_pos)) + list(np.array(self.right_hand_pos)))
                self.vel_traj.append(self.base_vel_yaw)
                self.eef_contact_traj.append(
                    [self.leftFoot_isInContact, self.rightFoot_isInContact])
                self.com_traj.append(self.calCOMPos())
                leftContactInfo = self._p.getContactPoints(
                    self.robot, self.plane, self.jointIdx["leftAnkleRoll"], -1)
                rightContactInfo = self._p.getContactPoints(
                    self.robot, self.plane, self.jointIdx["rightAnkleRoll"], -1)
                self.foot_force.append([leftContactInfo, rightContactInfo])

        self._observation = self.get_observation()
        reward, reward_info = copy.copy(self.getReward())
        done = copy.copy(self.checkFall())
        self.prev_action = copy.copy(self.action)
        return copy.copy(self._observation), reward, done, reward_info

    def render(self, mode='human', close=False, distance=3, yaw=0, pitch=-30, roll=0, high_res=False):
        scale = 2 if not high_res else 1
        width = 1600//scale
        height = 900//scale

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[1, 0, 1],
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )

        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1,
            farVal=100.0,
        )

        (_, _, px, _, _) = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_TINY_RENDERER
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def reset(self, loadObject=True, start_frame=None, isTest=False, base_pos_nom=None):
        return self._reset(loadObject=loadObject, start_frame=start_frame, isTest=isTest, base_pos_nom=base_pos_nom)

    def _reset(self, base_pos_nom=None, base_orn_nom=None, fixed_base=False, q_nom=None, loadObject=True, start_frame=None, isTest=False):
        self.door_pos_init = 0

        if self.imitate_motion:
            if self.imitateWalking:
                self.human_motion.reset(index=1)  # walking
            else:
                self.human_motion.reset(index=0)  # kicking
            if start_frame is not None:
                self.human_motion.count = start_frame
            else:
                self.human_motion.random_count()  # random reset
            self.q_nom_list = self.get_first_frame()
        self.start_count = copy.copy(self.human_motion.count)
        self.stabilised = False
        self.door_is_open = False
        self.stop_gait = False
        self.box_grasped = False
        self.time = 0

        start_base_pos = [
            [3.697767643977121210e-01, 6.114491409162276653e-03],
            [5.778334140402680008e-01, -1.676779507972906424e-01], ]

        table_base_pos = [
            [2.440007581368520029e+00, -9.801258393216735199e-01],
        ]

        door_base_pos = [
            [3.223393539726230106e+00, -1.841788825117443462e+00],
            [3.583969711045512874e+00, -2.062605934461614332e+00],
            [4.138767767870523251e+00, -1.864001689375232251e+00],
            [4.565460641154535537e+00, -1.868349563100097255e+00],
        ]

        self.base_pos_spawn_options = []
        """Bad coding, but higher weights on start at beginning: 8 are start, 2 are at table, 4 are at door"""
        # 57% of beginning
        # 14% of table
        # 29% of door
        self.base_pos_spawn_options.extend(start_base_pos)  # 2 entries
        self.base_pos_spawn_options.extend(start_base_pos)  # 2 entries
        self.base_pos_spawn_options.extend(start_base_pos)  # 2 entries
        self.base_pos_spawn_options.extend(start_base_pos)  # 2 entries
        self.base_pos_spawn_options.extend(table_base_pos)  # 1 entry
        self.base_pos_spawn_options.extend(table_base_pos)  # 1 entry
        self.base_pos_spawn_options.extend(door_base_pos)  # 4 entries

        if self.random_spawn and not isTest:
            random_offset = random.choice(self.base_pos_spawn_options)
            self.base_pos_spawn_offset[0] = random_offset[0]
            self.base_pos_spawn_offset[1] = random_offset[1]
        else:
            self.base_pos_spawn_offset[0] = 0.2
            self.base_pos_spawn_offset[1] = 0.0
        if isTest:
            seed = 123
        else:
            seed = int((time.time()*1e6) % 1e9)

        np.random.seed(seed=seed)

        self._p.resetSimulation()
        self._setupSimulation(base_pos_nom, base_orn_nom, fixed_base, q_nom)

        # action random init is set in setupSimulation
        self.action = self.q_nom_list if not self.random_joint_init else self.action_random_init

        self._envStepCounter = 0

        if self.goal_type == "fixed":
            self.pelvis_goal = self.pelvis_goal if self.pelvis_goal is not None else [
                0.0, 0.0, 1.175]
        elif self.goal_type == "random_fixed":
            self.pelvis_goal = np.random.uniform(
                self.pelvis_range_low, self.pelvis_range_high)
        elif self.goal_type == "moving_goal":
            self.pelvis_goal = np.random.uniform(
                [0.5, -self.goal_y_range/2, 1.175], [0.5, self.goal_y_range/2, 1.175])
        elif self.goal_type is None:
            self.pelvis_goal = [0.0, 0.0, 1.175]
        elif self.goal_type == "fixed_behind":
            self.pelvis_goal = [-2.0, 0.0, 1.175]
        else:
            print("Reach type %s not defined" % self.goal_type)
            assert 3 == 4, "Reach type not defined"

        self.time_last_move_goal = -1

        if self.learn_stand:
            self.pelvis_goal[2] = 1.06
        """New force duration"""

        if self.fixed_disturbance:
            time1 = 15
            time2 = 55
            time3 = 65
        else:
            time1 = np.random.uniform(2.5, 4.5) if np.random.random(
            ) < self.probability_of_push else 10000
            time2 = np.random.uniform(5.0, 7.0) if np.random.random(
            ) < self.probability_of_push else 10000
            time3 = np.random.uniform(7.5, 9.5) if np.random.random(
            ) < self.probability_of_push else 10000
        force_durations = [self.force_duration for i in range(3)]
        force_magnitude_disturbance_x = []
        force_magnitude_disturbance_y = []

        for duration in force_durations:
            if self.fixed_disturbance:
                total_impulse_magnitude = -self.maxImpulse
            else:
                total_impulse_magnitude = np.random.uniform(
                    -self.maxImpulse, self.maxImpulse)
            y_component = 0
            x_component = total_impulse_magnitude

            x_component /= duration
            y_component /= duration

            force_magnitude_disturbance_x.append(x_component)
            force_magnitude_disturbance_y.append(y_component)

        self.disturbance_time = [
                                [time1, time1+force_durations[0], force_magnitude_disturbance_x[0],
                                    force_magnitude_disturbance_y[0]],
                                [time2, time2+force_durations[1], force_magnitude_disturbance_x[1],
                                    force_magnitude_disturbance_y[1]],
                                [time3, time3+force_durations[2], force_magnitude_disturbance_x[2],
                                    force_magnitude_disturbance_y[2]],
        ]

        if self.final_goal_type is not None:

            self.loadAllObjects(loadObject)
            self.pelvis_goal = self.final_goal
        if self.visualise_goal and (self.goal_type is not None):
            self.pelvis_goal_object = self._p.loadURDF(
                self.dir_path + "/urdf/target.urdf",
                basePosition=self.pelvis_goal,
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True)

        if self.filter_action:
            for _ in range(1000):
                self.action = self.getFilteredAction(self.q_nom_list)
                self._observation = self.get_observation()
        else:
            self._observation = self.get_observation()

        return np.array(self._observation)

    def get_first_frame(self):
        assert self.useFullDOF, "Imitation requires full dof"
        if self.lock_upper_body:
            if self.imitatePitchOnly:
                retVal = self.q_nom_list
                if len(self.human_motion.ref_joint_angle()) == 8:
                    assert len(self.q_nom_list) == 8
                    retVal[0] = self.human_motion.ref_joint_angle()[0]
                    retVal[1] = self.human_motion.ref_joint_angle()[1]
                    retVal[2] = self.human_motion.ref_joint_angle()[2]
                    retVal[3] = self.human_motion.ref_joint_angle()[3]
                    retVal[4] = self.human_motion.ref_joint_angle()[4]
                    retVal[5] = self.human_motion.ref_joint_angle()[5]
                    retVal[6] = self.human_motion.ref_joint_angle()[6]
                    retVal[7] = self.human_motion.ref_joint_angle()[7]

                    return np.array(retVal)
                elif len(self.human_motion.ref_joint_angle()) == 6:
                    assert len(self.q_nom_list) == 8
                    retVal[1] = self.human_motion.ref_joint_angle()[0]
                    retVal[2] = self.human_motion.ref_joint_angle()[1]
                    retVal[3] = self.human_motion.ref_joint_angle()[2]
                    retVal[5] = self.human_motion.ref_joint_angle()[3]
                    retVal[6] = self.human_motion.ref_joint_angle()[4]
                    retVal[7] = self.human_motion.ref_joint_angle()[5]

                    return np.array(retVal)
                else:
                    assert 3 == 4, "Not defined"
            else:
                ref_q = copy.copy(self.human_motion.ref_joint_angle())
                return ref_q
        else:
            retVal = self.q_nom_list
            assert len(self.human_motion.ref_joint_angle()) == 8
            retVal[-8] = self.human_motion.ref_joint_angle()[0]
            retVal[-7] = self.human_motion.ref_joint_angle()[1]
            retVal[-6] = self.human_motion.ref_joint_angle()[2]
            retVal[-5] = self.human_motion.ref_joint_angle()[3]
            retVal[-4] = self.human_motion.ref_joint_angle()[4]
            retVal[-3] = self.human_motion.ref_joint_angle()[5]
            retVal[-2] = self.human_motion.ref_joint_angle()[6]
            retVal[-1] = self.human_motion.ref_joint_angle()[7]
            return np.array(retVal)

    def get_observation(self):
        return self.getExtendedObservation()

    def getExtendedObservation_reach(self):
        if self.initialised:
            if np.linalg.norm(np.array(self.right_reach_goal_door)-np.array(self.right_hand_pos)) < 0.05 or self.door_is_open:
                self.openDoor()
            else:
                self.closeDoor()
            self.right_reach_goal_door = np.array(
                self._p.getLinkState(self.door, 2)[0])

            """Box stuff"""
            self.box_pos, self.box_quat = self._p.getBasePositionAndOrientation(
                self.box)
            self.box_vel, self.box_ang_vel = self._p.getBaseVelocity(
                self.box)

            self.left_reach_goal_box = np.array(
                self._p.getLinkState(self.box, 0)[0])
            self.right_reach_goal_box = np.array(
                self._p.getLinkState(self.box, 1)[0])

            """Calculating gravity vector"""
            invBasePos, invBaseQuat = self._p.invertTransform(
                [0, 0, 0], self.box_quat)
            gravity = np.array([0, 0, -1])  # in world coordinates
            gravity_quat = self._p.getQuaternionFromEuler([0, 0, 0])
            gravityPosInBase, gravityQuatInBase = self._p.multiplyTransforms(
                invBasePos, invBaseQuat, gravity, gravity_quat)
            self.box_gravity = np.array(gravityPosInBase)

            """Contact"""
            leftContactInfo = self._p.getContactPoints(
                self.robot, self.box, self.jointIdx["leftElbowPitch"], -1)
            rightContactInfo = self._p.getContactPoints(
                self.robot, self.box, self.jointIdx["rightElbowPitch"], -1)

            self.leftHand_isInContact = (len(leftContactInfo) > 0)
            self.rightHand_isInContact = (len(rightContactInfo) > 0)

            if self.leftHand_isInContact + self.rightHand_isInContact:
                self.box_grasped = True
                if self.left_reach_goal_height < 0.97+0.15:
                    self.left_reach_goal_height += 0.01
                if self.right_reach_goal_height < 0.97+0.15:
                    self.right_reach_goal_height += 0.01

                self.left_reach_goal_box[2] = self.left_reach_goal_height
                self.right_reach_goal_box[2] = self.right_reach_goal_height

            self.leftHandContactForce = [0, 0, 0]
            self.rightHandContactForce = [0, 0, 0]
            if self.leftHand_isInContact:
                for info in leftContactInfo:
                    # contact normal of foot pointing towards plane
                    contactNormal = np.array(info[7])
                    contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                    contactNormalForce = np.array(info[9])
                    F_contact = np.array(contactNormal)*contactNormalForce
                    self.leftHandContactForce += F_contact

            if self.rightHand_isInContact:
                for info in rightContactInfo:
                    # contact normal of foot pointing towards plane
                    contactNormal = np.array(info[7])
                    contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                    contactNormalForce = np.array(info[9])
                    F_contact = np.array(contactNormal)*contactNormalForce
                    self.rightHandContactForce += F_contact
        else:
            self.left_reach_goal_box = [0, 0, 0]
            self.right_reach_goal_box = [0, 0, 0]
            self.right_reach_goal_door = [0, 0, 0]

        """Observation"""
        self._observation = list(
            self.getFilteredObservation(obs_info_type="upper"))

        del(self._observation[:9])

        self._observation = copy.copy(
            list(self.left_reach_goal_box)) + copy.copy(self._observation)
        # have to do in like this to guarantee that last entry is joint position
        self._observation = copy.copy(
            list(self.right_reach_goal_box)) + copy.copy(self._observation)

        # have to do in like this to guarantee that last entry is joint position
        self._observation = copy.copy(
            list(self.left_hand_vel)) + copy.copy(self._observation)
        # have to do in like this to guarantee that last entry is joint position
        self._observation = copy.copy(
            list(self.right_hand_vel)) + copy.copy(self._observation)

        if self.initialised:
            # have to do in like this to guarantee that last entry is joint position
            self._observation = copy.copy(
                list(self.box_pos)) + copy.copy(self._observation)
            # have to do in like this to guarantee that last entry is joint position
            self._observation = copy.copy(
                list(self.box_gravity)) + copy.copy(self._observation)
        else:
            # have to do in like this to guarantee that last entry is joint position
            self._observation = copy.copy(
                [0., 0, 0]) + copy.copy(self._observation)
            # have to do in like this to guarantee that last entry is joint position
            self._observation = copy.copy(
                [0., 0, -1]) + copy.copy(self._observation)

        self._observation = np.array(self._observation)

        return self._observation

    def getExtendedObservation(self):
        if not self.stop_gait:
            self.human_motion.index_count()  # tick up

        self._observation = list(
            self.getFilteredObservation(obs_info_type="lower"))
        if self.goal_type == "moving_goal":
            if self.time - self.time_last_move_goal > self.goal_update_time:
                self.time_last_move_goal = self.time
                x_increment = self.target_velocity[0]/self.PD_freq
                self.pelvis_goal[0] = self.pelvis_goal[0] + x_increment

                sign = np.random.choice(a=[-1., 0, 1])
                y_increment = np.random.uniform(
                    0, self.target_velocity[1])/self.PD_freq

                self.pelvis_goal[1] = self.pelvis_goal[1] + sign*y_increment
                self.pelvis_goal = np.array(self.pelvis_goal)

        if self.visualise_goal and (self.goal_type is not None):
            self._p.resetBasePositionAndOrientation(
                self.pelvis_goal_object, self.pelvis_goal, [0, 0, 0, 1])
        """Absolute base pos"""
        if self.goal_type is not None:

            if self.goal_as_vel:
                # have to do in like this to guarantee that last entry is joint position
                self._observation = copy.copy(
                    self.vel_target[:2]) + copy.copy(self._observation)
                if self.obs_use_pos:
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        list(self.base_pos)[:2]) + copy.copy(self._observation)
                if self.obs_use_yaw:
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        [self.base_orn[2]]) + copy.copy(self._observation)
            else:
                if self.obs_use_pos and self.obs_use_yaw:  # absolute goal pos
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        list(self.pelvis_goal)[:2]) + copy.copy(self._observation)
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        list(self.base_pos)[:2]) + copy.copy(self._observation)
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        [self.base_orn[2]]) + copy.copy(self._observation)
                elif self.obs_use_pos:  # absolute goal pos
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        list(self.pelvis_goal)[:2]) + copy.copy(self._observation)
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        list(self.base_pos)[:2]) + copy.copy(self._observation)
                elif self.obs_use_yaw:  # relative goal pos
                    relative_goal = list(
                        np.array(self.pelvis_goal) - np.array(self.base_pos))
                    self._observation = copy.copy(
                        relative_goal[:2]) + copy.copy(self._observation)
                    self._observation = copy.copy(
                        [self.base_orn[2]]) + copy.copy(self._observation)
                elif not (self.obs_use_pos and self.obs_use_yaw):  # relative goal pos
                    relative_goal = list(
                        np.array(self.pelvis_goal) - np.array(self.base_pos))
                    # have to do in like this to guarantee that last entry is joint position
                    self._observation = copy.copy(
                        relative_goal[:2]) + copy.copy(self._observation)
                else:
                    assert 3 == 4, "Case use_pos: %d, use_yaw:%d not defined" % (
                        self.obs_use_pos, self.obs_use_yaw)
        """Add imitation phase"""
        if self.imitate_motion:
            phase = self.human_motion.count/self.human_motion.dsr_length

            phase_sin = np.sin(2*np.pi*phase)
            phase_cos = np.cos(2*np.pi*phase)
            self._observation = copy.copy(
                [phase_sin, phase_cos]) + copy.copy(self._observation)

        """Add past action"""
        if self.obs_use_foot_past_action:
            self._observation = copy.copy(
                list(self.prev_action)) + copy.copy(self._observation)

        self._observation = np.array(self._observation)
        return self._observation

    def getReward(self):
        if self.imitate_motion:
            if self.imit_weights["imitation"]:
                reward_imitation, reward_imitation_info = self.imitation_reward()
            else:
                reward_imitation, reward_imitation_info = 0, {}
            if self.imit_weights["goal"]:
                reward_goal, reward_goal_info = self.task_reward()
            else:
                reward_goal, reward_goal_info = 0, {}

            reward = self.imit_weights["goal"]*reward_goal + \
                self.imit_weights["imitation"]*reward_imitation
            reward_info = {}
            reward_info.update(reward_imitation_info)
            reward_info.update(reward_goal_info)
            reward_info.update(
                {"reward_imitation": reward_imitation, "reward_goal": reward_goal})
            if self.print_reward_details:
                print("=====================================")
                print("Average rewards: ")
                for key in reward_info:
                    if key not in self.reward_dict_list:
                        self.reward_dict_list.update({key: [reward_info[key]]})
                    else:
                        self.reward_dict_list[key].append(reward_info[key])
                    if np.mean(np.array(self.reward_dict_list[key])):
                        print("%s: %.2f" % (key, np.mean(
                            np.array(self.reward_dict_list[key]))))
        else:
            reward, reward_info = self.stand_reward()

        return reward, reward_info

    def setLateralFriction(self, lateral_friction=1.5):

        foot_stiffness = 1e6
        foot_damping = 100
        self._p.changeDynamics(self.robot, self.jointIdx["leftAnkleRoll"], lateralFriction=lateral_friction,
                               contactStiffness=foot_stiffness, contactDamping=foot_damping)
        self._p.changeDynamics(self.robot, self.jointIdx["rightAnkleRoll"], lateralFriction=lateral_friction,
                               contactStiffness=foot_stiffness, contactDamping=foot_damping)
        plane_stiffness = 1e6
        plane_damping = 100
        self._p.changeDynamics(
            self.plane, -1, lateralFriction=lateral_friction,
            contactStiffness=plane_stiffness, contactDamping=plane_damping)

    def applyForce(self, force=[0, 0, 0], linkName="base"):

        if linkName == 'base':
            index = -1
        else:
            index = self.jointIdx[linkName]
        frame_flag = self._p.LINK_FRAME
        pos = [0, 0, 0]
        self._p.applyExternalForce(self.robot, index,
                                   forceObj=force,
                                   posObj=pos,  # [0, 0.0035, 0],
                                   flags=frame_flag)

    def imitation_reward(self):

        request_stand_config = self.learn_stand
        ref_joint_pos_dict = self.human_motion.ref_joint_angle_dict(
            request_stand_config=request_stand_config)
        ref_joint_vel_dict = self.human_motion.ref_joint_vel_dict()

        ref_eef_data = self.human_motion.get_eef_data()

        alpha = 1e-2

        """Joint imitation reward"""
        joint_pos_reward = 0
        joint_vel_reward = 0
        total_weight = 0

        if self.save_trajectories:
            self.q_imit_traj.append(ref_joint_pos_dict)
            self.eef_imit_traj.append(ref_eef_data)

        for key in self.imitating_joints:
            assert key in self.human_motion.available_joint_list, "%s not in available joint list of self.human_motion" % key
            weight = self.joint_weights[key]
            total_weight += weight
            index = self.jointIdx[key]
            joint_state = self.joint_states[key]
            joint_pos = joint_state[0]
            joint_vel = joint_state[1]

            ref_joint_pos = ref_joint_pos_dict[key]
            ref_joint_vel = ref_joint_vel_dict[key]

            joint_pos_err = (ref_joint_pos - joint_pos) / \
                (self.joint_imit_tolerance_dict[key] * math.pi / 180)
            joint_vel_err = (ref_joint_vel - joint_vel) / \
                (22.5 * math.pi / 180)

            joint_pos_reward += weight * \
                np.exp(np.log(alpha) * (joint_pos_err) ** 2)
            joint_vel_reward += weight * \
                np.exp(np.log(alpha) * (joint_vel_err) ** 2)

        joint_pos_reward /= total_weight
        joint_vel_reward /= total_weight

        """Foot pos imitation reward"""
        lfoot_pos = np.array(self.base_pos) - np.array(self.left_foot_pos)
        rfoot_pos = np.array(self.base_pos) - np.array(self.right_foot_pos)
        lfoot_pos_err = ref_eef_data["eef_lfoot_pos"] - lfoot_pos
        rfoot_pos_err = ref_eef_data["eef_rfoot_pos"] - rfoot_pos

        eef_lfoot_pos_x_reward = np.exp(-10 * (lfoot_pos_err[0]) ** 2)
        eef_lfoot_pos_y_reward = np.exp(-40 * (lfoot_pos_err[1]) ** 2)
        eef_lfoot_pos_z_reward = np.exp(-40 * (lfoot_pos_err[2]) ** 2)

        eef_rfoot_pos_x_reward = np.exp(-10 * (rfoot_pos_err[0]) ** 2)
        eef_rfoot_pos_y_reward = np.exp(-40 * (rfoot_pos_err[1]) ** 2)
        eef_rfoot_pos_z_reward = np.exp(-40 * (rfoot_pos_err[2]) ** 2)

        if self.imitate_only_vertical_eef:
            eef_lfoot_pos_reward = eef_lfoot_pos_z_reward
            eef_rfoot_pos_reward = eef_rfoot_pos_z_reward
        else:
            eef_lfoot_pos_reward = (
                eef_lfoot_pos_x_reward+eef_lfoot_pos_y_reward+eef_lfoot_pos_z_reward)/3
            eef_rfoot_pos_reward = (
                eef_rfoot_pos_x_reward+eef_rfoot_pos_y_reward+eef_rfoot_pos_z_reward)/3

        eef_foot_pos_reward = 0.5*eef_lfoot_pos_reward+0.5*eef_rfoot_pos_reward

        """Foot orientation imitation reward"""
        lfoot_orientation_err = ref_eef_data["eef_lfoot_orientation"][1] - \
            self.left_foot_orientation[1]
        rfoot_orientation_err = ref_eef_data["eef_rfoot_orientation"][1] - \
            self.right_foot_orientation[1]

        eef_lfoot_orientation_reward = np.exp(
            np.log(alpha) * ((lfoot_orientation_err)/(45*3.14/180.)) ** 2)
        eef_rfoot_orientation_reward = np.exp(
            np.log(alpha) * ((rfoot_orientation_err)/(45*3.14/180.)) ** 2)

        eef_foot_orientation_reward = 0.5*eef_lfoot_orientation_reward + \
            0.5*eef_rfoot_orientation_reward
        """Foot contact imitation"""
        if self.imitate_contact_type == 0:
            gait_phase = self.human_motion.count/self.human_motion.dsr_length
            gap = self.dsp_duration/2
            if gait_phase >= 0 and gait_phase < gap:
                if (self.leftFoot_isInContact == True):  # allow double contact
                    contact_term = 2
                else:
                    contact_term = 0
            elif gait_phase >= gap and gait_phase < 0.5:
                if (self.rightFoot_isInContact == True) and (self.leftFoot_isInContact == False):
                    contact_term = 2
                else:
                    contact_term = 0
            elif gait_phase >= 0.5 and gait_phase < (0.5+gap):
                if (self.leftFoot_isInContact == True):  # allow double contact
                    contact_term = 2
                else:
                    contact_term = 0
            elif gait_phase >= (0.5+gap) and gait_phase <= 1.0:
                if (self.rightFoot_isInContact == False) and (self.leftFoot_isInContact == True):
                    contact_term = 2
                else:
                    contact_term = 0
            else:
                contact_term = 0

            reward = (3*joint_pos_reward + joint_vel_reward)*10/(3+1)
            reward += contact_term

            reward_term = []
            reward_term = dict([
                ("imitation_contact_term", contact_term),
                ("imitation_joint_pos_reward", joint_pos_reward),
                ("imitation_joint_vel_reward", joint_vel_reward),

            ])

        elif self.imitate_contact_type == 1:
            lfoot_contact_err = abs(
                ref_eef_data["eef_lfoot_contact"] - self.leftFoot_isInContact)
            rfoot_contact_err = abs(
                ref_eef_data["eef_rfoot_contact"] - self.rightFoot_isInContact)
            if (lfoot_contact_err + rfoot_contact_err) < 1e-3:
                contact_term = 1
            elif (lfoot_contact_err + rfoot_contact_err) == 1:
                contact_term = 0
            elif(lfoot_contact_err + rfoot_contact_err) == 2:
                contact_term = 0
            else:
                raise ValueError("Shouldnt exist")

            reward = 10*(self.weight_imit_joint_pos_reward*joint_pos_reward
                         + self.weight_imit_eef_contact_reward*contact_term
                         + self.weight_imit_eef_pos_reward*eef_foot_pos_reward
                         + self.weight_imit_eef_orientation_reward*eef_foot_orientation_reward) /\
                (self.weight_imit_joint_pos_reward
                 + self.weight_imit_eef_contact_reward
                 + self.weight_imit_eef_pos_reward
                 + self.weight_imit_eef_orientation_reward)

            reward_term = []
            reward_term = dict([
                ("imitation_joint_pos_reward",
                 self.weight_imit_joint_pos_reward*joint_pos_reward),
                ("imitation_contact_reward",
                 self.weight_imit_eef_contact_reward*contact_term),
                ("imitation_foot_pos_reward",
                 self.weight_imit_eef_pos_reward*eef_foot_pos_reward),
                ("imitation_foot_orientation_reward",
                 self.weight_imit_eef_orientation_reward*eef_foot_orientation_reward),
            ])
        else:
            raise ValueError("Contact type %d not defined " %
                             self.imitate_contact_type)
        """Summing up"""

        return reward, reward_term

    def reach_reward(self, clamp=False):
        print("Reach reward unused now! Still exists in valkyrie reach env")
        return 0, {}

    def task_reward(self):
        """Position error"""

        alpha = 1e-2

        if self.goal_type is None:
            x_tolerance = 0.1
            y_tolerance = 0.1
            z_tolerance = 0.1
        else:
            x_tolerance = 5.0
            y_tolerance = 1.0
            z_tolerance = 0.1

        x_pos_err = self.pelvis_goal[0] - self.base_pos[0]
        y_pos_err = self.pelvis_goal[1] - self.base_pos[1]
        z_pos_err = self.pelvis_goal[2] - self.base_pos[2]

        x_pos_reward = math.exp(math.log(alpha)*(x_pos_err/x_tolerance)**2)
        y_pos_reward = math.exp(math.log(alpha)*(y_pos_err/y_tolerance)**2)
        z_pos_reward = math.exp(math.log(alpha)*(z_pos_err/z_tolerance)**2)
        """Velocity error"""
        if self.goal_type is None:
            x_vel_target = self.target_velocity[0]
            y_vel_target = 0
            z_vel_target = 0
            if self.allow_faster_vel:
                # allow faster than target vel movements
                x_vel_err = np.maximum(
                    x_vel_target - self.base_vel_yaw[0], 0.0)
            else:
                x_vel_err = x_vel_target-self.base_vel_yaw[0]

            y_vel_err = y_vel_target-self.base_vel_yaw[1]
            z_vel_err = z_vel_target-self.base_vel_yaw[2]
        else:
            vel_vector = np.array(
                self.pelvis_goal[:2]) - np.array(self.base_pos[:2])
            vel_vector_normalised = vel_vector/np.linalg.norm(vel_vector)

            x_vel_target = vel_vector_normalised[0]*self.target_velocity[0]
            y_vel_target = vel_vector_normalised[1]*self.target_velocity[0]
            assert abs(self.target_velocity[2]) < 1e-3
            z_vel_target = self.target_velocity[2]  # should be zero

            if self.allow_faster_vel:
                # allow faster than target vel movements
                x_vel_err = np.maximum(x_vel_target - self.base_vel[0], 0.0)
            else:
                x_vel_err = x_vel_target-self.base_vel[0]

            y_vel_err = y_vel_target-self.base_vel[1]
            z_vel_err = z_vel_target-self.base_vel[2]

        self.vel_target = copy.copy([x_vel_target, y_vel_target])

        alpha = 1e-3

        vx_tolerance = 1.0 if abs(x_vel_target) > 0.2 or (
            not self.tighter_tolerance_upon_reaching_goal) else 0.2
        vy_tolerance = 1.0
        vz_tolerance = 1.0

        x_vel_reward = math.exp(-10*(x_vel_err/vx_tolerance)**2)
        y_vel_reward = math.exp(-10*(y_vel_err/vy_tolerance)**2)
        z_vel_reward = math.exp(-10*(z_vel_err/vy_tolerance)**2)

        if self.print_reward_details:
            print("Vel error: [%.2f, %.2f, %.2f], target: [%.2f, %.2f, %.2f], is [%.2f, %.2f, %.2f]. Reward: [%.2f, %.2f, %.2f]"
                  % (x_vel_err, y_vel_err, z_vel_err, x_vel_target, y_vel_target, z_vel_target, self.base_vel_yaw[0], self.base_vel_yaw[1], self.base_vel_yaw[2],
                     x_vel_reward, y_vel_reward, z_vel_reward))

        """Orientation rewards"""

        # gravity vector error
        if self.goal_type is None:
            gravity_vector_tar = np.array([0, 0, -1])
            base_gravity_vector_err = np.linalg.norm(
                gravity_vector_tar-self.base_gravity_vector)

            gravity_reward = math.exp(
                math.log(alpha)*(base_gravity_vector_err/1.4)**2)
        else:

            # character orientation to goal
            orientation_to_goal_vec = np.array(
                [vel_vector_normalised[0], vel_vector_normalised[1], 0])
            # unit character to ball vector in robot frame
            orientation_to_goal_vec.resize(1, 3)
            orientation_to_goal_yaw = np.transpose(
                self.Rz_i @ orientation_to_goal_vec.transpose())

            orientation_to_goal_vec_tar = np.array([1, 0, 0])
            orientation_to_goal_vec_err = np.linalg.norm(
                (orientation_to_goal_vec_tar-orientation_to_goal_yaw)[0][:2])  # the 3rd element (z) is zero all the time
            gravity_reward = math.exp(
                math.log(alpha)*(orientation_to_goal_vec_err/1.4)**2)

        if self.save_trajectories:
            if self.goal_type is not None:
                self.pelvis_goal_traj.append(copy.copy(self.pelvis_goal[:2]))
                self.pelvis_pos_traj.append(copy.copy(self.base_pos[:2]))
                self.vel_goal_traj.append(
                    [x_vel_target, y_vel_target, z_vel_target])
                self.grav_traj.append(orientation_to_goal_yaw[0])
                self.gravity_goal_traj.append([1, 0, 0])

        target_pitch = 0

        lfoot_pitch_err = target_pitch - self.left_foot_orientation[1]
        rfoot_pitch_err = target_pitch - self.right_foot_orientation[1]

        pitch_tolerance = 45*3.14/180.

        lfoot_pitch_reward = math.exp(
            math.log(alpha)*(lfoot_pitch_err / pitch_tolerance)**2)
        rfoot_pitch_reward = math.exp(
            math.log(alpha)*(rfoot_pitch_err / pitch_tolerance)**2)

        foot_pitch_reward = (lfoot_pitch_reward+rfoot_pitch_reward)/2

        """Force distribution rewards"""
        force_targ = -self.total_mass*self.g/2.0
        left_foot_force_err = force_targ - \
            self.leftContactForce[2]  # Z contact force
        right_foot_force_err = force_targ-self.rightContactForce[2]
        left_foot_force_reward = math.exp(
            math.log(alpha)*(left_foot_force_err/force_targ)**2)
        right_foot_force_reward = math.exp(
            math.log(alpha)*(right_foot_force_err/force_targ)**2)

        if self.regularise_action:
            velocity_error = 0
            torque_error = 0
            for idx, key in enumerate(self.controlled_joints):
                velocity_error += (self.joint_velocity[idx] /
                                   self.v_max[key])**2
                torque_error += (self.joint_torques[idx] / self.u_max[key])**2
            velocity_error /= len(self.controlled_joints)
            torque_error /= len(self.controlled_joints)

            joint_vel_reward = math.exp(math.log(alpha)*velocity_error)
            joint_torque_reward = math.exp(math.log(alpha)*torque_error)
        else:
            joint_vel_reward = 0
            joint_torque_reward = 0

        """Foot clearance reward"""
        foot_z = 0.15  # swing foot clearance

        left_foot_vel_norm = np.linalg.norm(self.left_foot_vel)
        lfoot_pos_err = self.left_foot_pos[2] - foot_z
        # allow foot height to be higher than target foot height
        lfoot_pos_err = np.minimum(lfoot_pos_err, 0.0)

        # trade off allows foot to be fast when high in air, but needs to be slow when foot clearance error large (close to ground)
        lfoot_err = ((lfoot_pos_err)/foot_z)**2*left_foot_vel_norm
        lfoot_clearance = math.exp(
            math.log(alpha) * lfoot_err) if not self.leftFoot_isInContact else 0.0

        right_foot_vel_norm = np.linalg.norm(self.right_foot_vel)
        rfoot_pos_err = self.right_foot_pos[2] - foot_z
        # allow foot height to be higher than target foot height
        rfoot_pos_err = np.minimum(rfoot_pos_err, 0.0)

        rfoot_err = ((rfoot_pos_err)/foot_z)**2*right_foot_vel_norm
        rfoot_clearance = math.exp(
            math.log(alpha) * rfoot_err) if not self.rightFoot_isInContact else 0.0

        foot_clearance_reward = (lfoot_clearance+rfoot_clearance)/2.0

        """Foot slippage"""
        slippage_version = 1
        if slippage_version == 0:
            lfoot_slippage_err = -left_foot_vel_norm/1.0
            lfoot_slippage = math.exp(
                math.log(alpha) * lfoot_slippage_err**2) if self.leftFoot_isInContact else 0.0

            rfoot_slippage_err = -right_foot_vel_norm/1.0
            rfoot_slippage = math.exp(
                math.log(alpha) * rfoot_slippage_err**2) if self.rightFoot_isInContact else 0.0

            foot_slippage_reward = (lfoot_slippage + rfoot_slippage)/2.0
        else:
            foot_slippage_reward = 0
            if self.leftFoot_isInContact:
                foot_slippage_reward += 1 if self.leftFoot_contactPoints == 4 else 0
            if self.rightFoot_isInContact:
                foot_slippage_reward += 1 if self.rightFoot_contactPoints == 4 else 0

            foot_slippage_reward = foot_slippage_reward/(self.leftFoot_isInContact + self.rightFoot_isInContact) if (
                self.leftFoot_isInContact + self.rightFoot_isInContact) > 0 else 0

        weight_x_pos_reward = self.weight_x_pos_reward  # 2.0
        weight_y_pos_reward = self.weight_y_pos_reward  # 2.0
        weight_z_pos_reward = self.weight_z_pos_reward  # 2.0
        weight_x_vel_reward = self.weight_x_vel_reward  # 6.0
        weight_y_vel_reward = self.weight_y_vel_reward  # 2.0
        weight_z_vel_reward = self.weight_z_vel_reward  # 2.0
        weight_torso_pitch_reward = self.weight_torso_pitch_reward  # 0.5
        weight_pelvis_pitch_reward = self.weight_pelvis_pitch_reward  # 0.5
        weight_left_foot_force_reward = self.weight_left_foot_force_reward  # 1.0
        weight_right_foot_force_reward = self.weight_right_foot_force_reward  # 1.0
        weight_joint_vel_reward = self.weight_joint_vel_reward  # 1.0
        weight_joint_torque_reward = self.weight_joint_torque_reward  # 1.0
        weight_foot_clearance_reward = self.weight_foot_clearance_reward
        weight_foot_slippage_reward = self.weight_foot_slippage_reward
        weight_foot_pitch_reward = self.weight_foot_pitch_reward
        weight_foot_contact_reward = self.weight_foot_contact_reward
        weight_gravity_reward = self.weight_gravity_reward
        weight_contact_penalty = self.weight_contact_penalty

        contact_penalty = -self.contactWithObject() if weight_contact_penalty else 0.0
        if self.learn_stand:
            foot_contact_reward = 1 if (
                self.leftFoot_isInContact and self.rightFoot_isInContact) else 0
        else:
            foot_contact_reward = 0 if (
                self.leftFoot_isInContact or self.rightFoot_isInContact) else -1

        reward = (
            weight_x_pos_reward * x_pos_reward
            + weight_y_pos_reward * y_pos_reward
            + weight_z_pos_reward * z_pos_reward
            + weight_x_vel_reward * x_vel_reward
            + weight_y_vel_reward * y_vel_reward
            + weight_z_vel_reward * z_vel_reward
            + weight_gravity_reward * gravity_reward
            + weight_left_foot_force_reward * left_foot_force_reward
            + weight_right_foot_force_reward * right_foot_force_reward
            + weight_joint_vel_reward * joint_vel_reward
            + weight_joint_torque_reward * joint_torque_reward
            + weight_foot_clearance_reward*foot_clearance_reward
            + weight_foot_slippage_reward*foot_slippage_reward
            + weight_foot_pitch_reward*foot_pitch_reward
            + foot_contact_reward*weight_foot_contact_reward
            + weight_contact_penalty*contact_penalty
        ) \
            * 10 / (weight_x_pos_reward + weight_y_pos_reward + weight_z_pos_reward
                    + weight_x_vel_reward + weight_y_vel_reward + weight_z_vel_reward
                    + weight_gravity_reward
                    + weight_left_foot_force_reward + weight_right_foot_force_reward
                    + weight_joint_vel_reward + weight_joint_torque_reward
                    + weight_foot_clearance_reward + weight_foot_slippage_reward
                    + weight_foot_pitch_reward + weight_foot_contact_reward
                    + weight_contact_penalty)

        reward_term = dict([
            ("x_pos_reward",            weight_x_pos_reward*x_pos_reward),
            ("y_pos_reward",            weight_y_pos_reward*y_pos_reward),
            ("z_pos_reward",            weight_z_pos_reward*z_pos_reward),
            ("x_vel_reward",            weight_x_vel_reward*x_vel_reward),
            ("y_vel_reward",            weight_y_vel_reward*y_vel_reward),
            ("z_vel_reward",            weight_z_vel_reward*z_vel_reward),
            ("gravity_reward",          weight_gravity_reward*gravity_reward),
            ("joint_vel_reward",        weight_joint_vel_reward*joint_vel_reward),
            ("joint_torque_reward",     weight_joint_torque_reward*joint_torque_reward),
            ("foot_clearance_reward",
             weight_foot_clearance_reward*foot_clearance_reward),
            ("foot_slippage_reward",
             weight_foot_slippage_reward*foot_slippage_reward),
            ("foot_contact_reward",     weight_foot_contact_reward*foot_contact_reward),
            ("contact_penalty", weight_contact_penalty*contact_penalty)
        ])

        return reward, reward_term

    def stand_reward(self):

        return 0, {}

    def resetJointStates(self, q_nom=None):
        """Uncomment for setting desired pos"""

        if q_nom is None:
            q_nom = self.q_nom
        else:
            # replace nominal joint angle with target joint angle
            temp = dict(self.q_nom)
            for key, value in q_nom.items():
                temp[key] = value
            q_nom = dict(temp)

        self.random_q_nom = {}
        for jointName in q_nom:
            if self.random_joint_init and jointName in self.controlled_joints:
                _range = 20*3.14/180
                val = np.random.uniform(-_range, _range)
                self.random_q_nom.update(
                    {jointName: self.q_nom[jointName]+val})
            else:
                val = 0

            self._p.resetJointState(self.robot,
                                    self.jointIdx[jointName],
                                    targetValue=q_nom[jointName]+val,
                                    targetVelocity=0)
        if self.random_joint_init:
            self.action_random_init = []
            for jointName in self.controlled_joints:
                self.action_random_init.append(self.random_q_nom[jointName])

    def _setupSimulation(self, base_pos_nom=None, base_orn_nom=None, fixed_base=False, q_nom=None):
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom

        if self.imitate_motion:
            if self.human_motion.stand_config is not None:
                probability_for_stand = 0. if self.learn_stand else 0.0
                request_stand_config = np.random.random() < probability_for_stand
            else:
                request_stand_config = False
            if request_stand_config:
                base_pos_nom[2] = self.pelvis_goal[2]+0.01

            q_nom = self.human_motion.ref_joint_angle_dict(
                request_stand_config=request_stand_config)

        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0, 0, -self.g)
        self._p.setTimeStep(self._dt)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        plane_urdf = self.dir_path + "/urdf/plane.urdf"

        self.plane = self._p.loadURDF(plane_urdf, basePosition=[
                                      0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)

        if self.useFullDOF:
            valkyrie_urdf = self.dir_path + "/urdf/valkyrie_full_dof.urdf" if self.urdf_version == 0 else self.dir_path + \
                "/urdf/valkyrie_full_dof_old.urdf"

            if self.replace_foot == 1:
                valkyrie_urdf = self.dir_path + "/urdf/valkyrie_no_lfoot.urdf"
            elif self.replace_foot == 2:
                valkyrie_urdf = self.dir_path + "/urdf/valkyrie_no_rfoot.urdf"

        else:
            valkyrie_urdf = self.dir_path + "/urdf/valkyrie_reduced_fixed.urdf"

        basePos = np.array(
            base_pos_nom)+np.array(self.base_pos_spawn_offset)*self.applySpawnOffset
        self.robot = self._p.loadURDF(fileName=valkyrie_urdf,
                                      basePosition=basePos,
                                      baseOrientation=base_orn_nom,
                                      flags=self._p.URDF_USE_INERTIA_FROM_FILE,
                                      useFixedBase=self.fixed_base,
                                      )

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        jointIds = []

        for j in range(self._p.getNumJoints(self.robot)):
            info = self._p.getJointInfo(self.robot, j)
            jointName = info[1].decode("utf-8")

            linearDamping = 1 if "Shoulder" not in jointName or "Elbow" not in jointName else 0
            angularDamping = 0 if "Shoulder" not in jointName or "Elbow" not in jointName else 0

            self._p.changeDynamics(
                self.robot, j, linearDamping=linearDamping, angularDamping=angularDamping)

            jointType = info[2]
            if (jointType == self._p.JOINT_REVOLUTE):
                jointIds.append(j)

            self.jointIdx.update({jointName: info[0]})
            self.jointNameIdx.update({info[0]: jointName})

            link_name = info[-5].decode("utf-8")
            self.linkIdx.update({link_name: info[0]})
            self.linkNameIdx.update({info[0]: link_name})

        self.jointIds = []
        for name in self.controlled_joints:
            _id = self.jointIdx[name]
            if _id in jointIds:
                self.jointIds.append(_id)

        if self.margin > 90*3.14/180:

            for joint in self.controlled_joints:
                info = self._p.getJointInfo(self.robot, self.jointIdx[joint])
                self.joint_limits_low.update({joint: (info[8])})
                self.joint_limits_high.update({joint: (info[9])})
        else:
            if self.imitate_motion:
                for joint in self.controlled_joints:
                    if joint not in self.joint_limits_low.keys():
                        info = self._p.getJointInfo(
                            self.robot, self.jointIdx[joint])
                        self.joint_limits_low.update({joint: (info[8])})
                        self.joint_limits_high.update({joint: (info[9])})

        self.jointLowerLimit = []
        self.jointUpperLimit = []
        for jointName in self.controlled_joints:
            self.jointLowerLimit.append(self.joint_limits_low[jointName])
            self.jointUpperLimit.append(self.joint_limits_high[jointName])
        self.joint_increment = (
            np.array(self.jointUpperLimit) - np.array(self.jointLowerLimit)) / (25 * 10)

        self.resetJointStates(q_nom)
        self.setLateralFriction(lateral_friction=self.lateral_friction)

    def getObservation(self, obs_info_type=None):
        self.base_pos, self.base_quat = self._p.getBasePositionAndOrientation(
            self.robot)
        self.base_vel, self.base_orn_vel = self._p.getBaseVelocity(self.robot)

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
        self.chest_link_state = self._p.getLinkState(
            self.robot, self.jointIdx['torsoRoll'], computeLinkVelocity=1)

        self.linkStates = self._p.getLinkStates(
            self.robot, queried_indices, computeLinkVelocity=True)

        self.head_pos = self.linkStates[0][0]
        self.left_elbow_pos = self.linkStates[1][0]
        self.right_elbow_pos = self.linkStates[2][0]
        self.left_hand_pos = self.linkStates[3][0]
        self.right_hand_pos = self.linkStates[4][0]

        self.torso_pos = self.linkStates[5][0]
        self.left_knee_pos = self.linkStates[6][0]
        self.right_knee_pos = self.linkStates[7][0]
        self.left_foot_pos = self.linkStates[8][0]
        self.right_foot_pos = self.linkStates[9][0]

        # penultimate entry is linear velocity
        assert self.linkStates[0][6] == self.linkStates[0][-2]
        self.head_vel = self.linkStates[0][6]
        self.left_elbow_vel = self.linkStates[1][6]
        self.right_elbow_vel = self.linkStates[2][6]
        self.left_hand_vel = self.linkStates[3][6]
        self.right_hand_vel = self.linkStates[4][6]
        self.torso_vel = self.linkStates[5][6]
        self.left_knee_vel = self.linkStates[6][6]
        self.right_knee_vel = self.linkStates[7][6]
        self.left_foot_vel = self.linkStates[8][6]
        self.right_foot_vel = self.linkStates[9][6]

        self.head_quat = self.linkStates[0][1]
        self.left_elbow_quat = self.linkStates[1][1]
        self.right_elbow_quat = self.linkStates[2][1]
        self.left_hand_quat = self.linkStates[3][1]
        self.right_hand_quat = self.linkStates[4][1]
        self.torso_quat = self.linkStates[5][1]
        self.left_knee_quat = self.linkStates[6][1]
        self.right_knee_quat = self.linkStates[7][1]
        self.left_foot_quat = self.linkStates[8][1]
        self.right_foot_quat = self.linkStates[9][1]

        self.head_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[0][1])
        self.left_elbow_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[1][1])
        self.right_elbow_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[2][1])
        self.left_hand_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[3][1])
        self.right_hand_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[4][1])
        self.torso_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[5][1])
        self.left_knee_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[6][1])
        self.right_knee_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[7][1])
        self.left_foot_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[8][1])
        self.right_foot_orientation = self._p.getEulerFromQuaternion(
            self.linkStates[9][1])

        leftContactInfo = self._p.getContactPoints(
            self.robot, self.plane, self.jointIdx["leftAnkleRoll"], -1)
        rightContactInfo = self._p.getContactPoints(
            self.robot, self.plane, self.jointIdx["rightAnkleRoll"], -1)

        contact_points_required = 4 if self.require_full_contact_foot else 0
        self.leftFoot_isInContact = (
            len(leftContactInfo) > contact_points_required)
        self.leftFoot_contactPoints = len(leftContactInfo)
        self.rightFoot_isInContact = (
            len(rightContactInfo) > contact_points_required)
        self.rightFoot_contactPoints = len(rightContactInfo)

        self.leftContactForce = [0, 0, 0]
        self.rightContactForce = [0, 0, 0]
        if self.leftFoot_isInContact:
            for info in leftContactInfo:
                # contact normal of foot pointing towards plane
                contactNormal = np.array(info[7])
                contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                contactNormalForce = np.array(info[9])
                F_contact = np.array(contactNormal)*contactNormalForce
                self.leftContactForce += F_contact

        if self.rightFoot_isInContact:
            for info in rightContactInfo:
                # contact normal of foot pointing towards plane
                contactNormal = np.array(info[7])
                contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                contactNormalForce = np.array(info[9])
                F_contact = np.array(contactNormal)*contactNormalForce
                self.rightContactForce += F_contact

        for name in self.controlled_joints:
            _id = self.jointIdx[name]

            self.joint_states.update(
                {self.jointNameIdx[_id]: self._p.getJointState(self.robot, _id)})

        """Observation"""
        observation = []
        """Yaw adjusted base linear velocity"""
        self.base_orn = self._p.getEulerFromQuaternion(self.base_quat)
        Rz = rotZ(self.base_orn[2])
        self.Rz_i = np.linalg.inv(Rz)
        base_vel = np.array(self.base_vel)
        base_vel.resize(1, 3)
        # base velocity in adjusted yaw frame
        self.base_vel_yaw = np.transpose(self.Rz_i @ base_vel.transpose())[0]

        base_vel_yaw_list = list(copy.copy(self.base_vel_yaw))
        observation.extend(copy.copy(base_vel_yaw_list))

        """Calculating gravity vector"""
        invBasePos, invBaseQuat = self._p.invertTransform(
            [0, 0, 0], self.base_quat)
        gravity = np.array([0, 0, -1])  # in world coordinates
        gravity_quat = self._p.getQuaternionFromEuler([0, 0, 0])
        gravityPosInBase, gravityQuatInBase = self._p.multiplyTransforms(
            invBasePos, invBaseQuat, gravity, gravity_quat)
        self.base_gravity_vector = np.array(gravityPosInBase)
        observation.extend(copy.copy(self.base_gravity_vector))

        """base angular velocity"""
        R = quat_to_rot(self.base_quat)
        self.R_i = np.linalg.inv(R)

        base_orn_vel = np.array(self.base_orn_vel)
        base_orn_vel.resize(1, 3)
        base_orn_vel_base = np.transpose(self.R_i @ base_orn_vel.transpose())

        base_orn_vel_base_list = list(copy.copy(base_orn_vel_base[0]))

        observation.extend(copy.copy(base_orn_vel_base_list))

        if obs_info_type is not None:
            if obs_info_type == "upper":
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
            elif obs_info_type == "lower":
                joint_info = [
                    "rightHipRoll",
                    "rightHipPitch",
                    "rightKneePitch",
                    "rightAnklePitch",
                    "leftHipRoll",
                    "leftHipPitch",
                    "leftKneePitch",
                    "leftAnklePitch",
                ]
            else:
                print("%s not defined" % self.obs_info_type)
        else:
            joint_info = self.controlled_joints

        self.joint_position = [self.joint_states[jointName][0]
                               for jointName in joint_info]
        self.joint_velocity = [self.joint_states[jointName][1]
                               for jointName in self.controlled_joints]
        self.joint_torques = [self.joint_states[jointName][-1]
                              for jointName in self.controlled_joints]

        if self.obs_use_foot_force:
            observation.extend(
                [self.leftContactForce[2], self.rightContactForce[2]])

        if self.obs_use_foot_pose:
            observation.extend(list(self.left_foot_pos) +
                               [self.left_foot_orientation[1]])
            observation.extend(list(self.right_foot_pos) +
                               [self.right_foot_orientation[1]])

        """Joint position (has to be last for smooth loss and split operator)"""
        observation.extend(copy.copy(self.joint_position))
        """Return observation"""
        observation = np.array(observation)
        return observation

    def getFilteredObservation(self, obs_info_type=None):
        observation = self.getObservation(obs_info_type=obs_info_type)
        if self.filter_observation:

            observation_filtered = self.state_filter_method.applyFilter(
                observation)

            observation = copy.copy(observation_filtered)

        return observation

    def rotX(self, theta):
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)]])
        return R

    def rotY(self, theta):
        R = np.array([
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)]])
        return R

    def rotZ(self, theta):
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0]])
        return R

    def transform(self, qs):  # transform quaternion into rotation matrix
        qx = qs[0]
        qy = qs[1]
        qz = qs[2]
        qw = qs[3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = np.empty([3, 3])
        m[0, 0] = 1.0 - (yy + zz)
        m[0, 1] = xy - wz
        m[0, 2] = xz + wy
        m[1, 0] = xy + wz
        m[1, 1] = 1.0 - (xx + zz)
        m[1, 2] = yz - wx
        m[2, 0] = xz - wy
        m[2, 1] = yz + wx
        m[2, 2] = 1.0 - (xx + yy)

        return m

    def euler_to_quat(self, roll, pitch, yaw):  # rad
        cy = np.cos(yaw*0.5)
        sy = np.sin(yaw*0.5)
        cr = np.cos(roll*0.5)
        sr = np.sin(roll*0.5)
        cp = np.cos(pitch*0.5)
        sp = np.sin(pitch*0.5)

        w = cy*cr*cp+sy*sr*sp
        x = cy*sr*cp-sy*cr*sp
        y = cy*cr*sp+sy*sr*cp
        z = sy*cr*cp-cy*sr*sp

        return [x, y, z, w]

    def checkFall(self):
        fall = False

        if not self.fixed_base:
            if self.base_pos[2] <= 1.0 or self.base_pos[2] > 1.4:
                fall = True

            if self.terminate_if_flight_phase:
                if self.time > 1.0:
                    fall = not (
                        self.leftFoot_isInContact or self.rightFoot_isInContact) or fall

            if self.terminate_if_not_double_support:
                fall = not (
                    self.leftFoot_isInContact and self.rightFoot_isInContact) or fall

            if self.terminate_if_pelvis_out_of_range:
                yaw_tolerance = 30*3.14/180
                y_tolerance = 0.3  # roughly half the width of distance between feet
                if abs(self.base_orn[2]) > yaw_tolerance:
                    fall = True
                if abs(self.base_pos[1]) > y_tolerance:
                    fall = True

        return fall

    def set_pos(self, action):
        assert len(action) == len(self.jointIds), "Action len: %d, joint id len: %d" % (
            len(action), len(self.jointIds))
        if self.gravity_compensation:
            if type(action) == np.ndarray:
                action = list(action)  # segfault if action is not a list
            torques_gravity_compensation = self._p.calculateInverseDynamics(
                self.robot, objPositions=action, objVelocities=[0.]*len(action), objAccelerations=[0.]*len(action))
        joint_states = self._p.getJointStates(self.robot, self.jointIds)
        torques = []
        v_max = []

        for n, _id in enumerate(self.jointIds):
            pos = joint_states[n][0]
            vel = joint_states[n][1]
            name = self.jointNameIdx[_id]
            pos_ref = action[n]
            P_u = self.Kp[name] * (pos_ref - pos)
            D_u = self.Kd[name] * (0-vel)
            control_torque = P_u+D_u
            if self.gravity_compensation:
                control_torque += torques_gravity_compensation[n]
            v_max.append(self.v_max[name])
            control_torque = np.clip(
                control_torque, -self.u_max[name], self.u_max[name])
            torques.append(control_torque)
        torques = np.array(torques)

        self._p.setJointMotorControlArray(
            self.robot, self.jointIds, targetVelocities=np.sign(
                torques) * v_max,
            forces=np.abs(torques),
            controlMode=self._p.VELOCITY_CONTROL,
        )
        if self.save_trajectories:
            self.pd_target.append(copy.copy(action))
            self.pd_value.append([joint_states[n][0]
                                  for n in range(len(self.jointIds))])

    def getPelvisInLeftFootFrame(self):
        return np.array(self.base_pos) - np.array(self.left_foot_pos)

    def getJointPositionForHands(self, lhand_pos=None, rhand_pos=None):

        ik_joints = [
            "left_hand",
            "right_hand",
        ]

        ik_joint_ids = []

        if lhand_pos is None:
            lhand_pos = copy.copy(self.left_reach_goal_box)
        if rhand_pos is None:
            rhand_pos = copy.copy(self.right_reach_goal_box)

        for ik_joint in ik_joints:
            ik_joint_ids.append(self.jointIdx[ik_joint])

        q = self._p.calculateInverseKinematics2(self.robot, ik_joint_ids,
                                                [lhand_pos, rhand_pos])

        return np.array(q)[:8]

    def getJointPositionForPelvis(self, pelvis_pos):

        if "left_foot_start" not in self.__dict__:
            self.left_foot_start = self.left_foot_pos
        if "right_foot_start" not in self.__dict__:
            self.right_foot_start = self.right_foot_pos

        if "left_knee_start" not in self.__dict__:
            self.left_knee_start = self.left_knee_pos
        if "right_knee_start" not in self.__dict__:
            self.right_knee_start = self.right_knee_pos

        if "left_elbow_start" not in self.__dict__:
            self.left_elbow_start = self.left_elbow_pos
        if "right_elbow_start" not in self.__dict__:
            self.right_elbow_start = self.right_elbow_pos
        if "left_hand_start" not in self.__dict__:
            self.left_hand_start = self.left_hand_pos
        if "right_hand_start" not in self.__dict__:
            self.right_hand_start = self.right_hand_pos

        if "torso_start" not in self.__dict__:
            self.torso_start = self.torso_pos
        if "head_start" not in self.__dict__:
            self.head_start = self.head_pos

        # given is pelvis desired in left foot frame
        pelvis_pos_pelvis_frame = np.array(
            pelvis_pos) - np.array(self.left_foot_start)

        left_foot_pos = self.left_foot_start - pelvis_pos_pelvis_frame
        right_foot_pos = self.right_foot_start - pelvis_pos_pelvis_frame

        left_elbow_pos = self.left_elbow_start - pelvis_pos_pelvis_frame
        right_elbow_pos = self.right_elbow_start - pelvis_pos_pelvis_frame
        left_hand_pos = self.left_hand_start - pelvis_pos_pelvis_frame
        right_hand_pos = self.right_hand_start - pelvis_pos_pelvis_frame

        torso_pos = self.torso_start - pelvis_pos_pelvis_frame
        head_pos = self.head_start - pelvis_pos_pelvis_frame

        ik_joints = [
            "head_imu_joint",
            "leftElbowPitch",
            "rightElbowPitch",
            "left_hand",
            "right_hand",
            "torsoYaw",
            "leftAnklePitch",
            "rightAnklePitch",
        ]
        ik_joint_ids = []
        for ik_joint in ik_joints:
            ik_joint_ids.append(self.jointIdx[ik_joint])

        q = self._p.calculateInverseKinematics2(self.robot, ik_joint_ids,
                                                [head_pos, left_elbow_pos, right_elbow_pos, left_hand_pos, right_hand_pos, torso_pos, left_foot_pos, right_foot_pos])
        return q

    def getJacobian(self, q=None, action_vector=None, qdot=None, qddot=None, localPosition=None):
        if q is None:
            q = self.joint_position
        if qdot is None:
            qdot = [0.]*len(q)
        if qddot is None:
            qddot = [0.]*len(q)
        if localPosition is None:
            localPosition = [0, 0, 0]  # using CoM of link as position
        _jacobian_lfoot = self._p.calculateJacobian(
            self.robot, self.jointIdx["leftAnklePitch"], localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        _jacobian_rfoot = self._p.calculateJacobian(
            self.robot, self.jointIdx["rightAnklePitch"], localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        _jacobian_pelvis = self._p.calculateJacobian(
            self.robot, -1, localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        jacobian_lfoot = np.vstack((_jacobian_lfoot[0], _jacobian_lfoot[1]))
        jacobian_rfoot = np.vstack((_jacobian_rfoot[0], _jacobian_rfoot[1]))
        jacobian_pelvis = np.vstack((_jacobian_pelvis[0], _jacobian_pelvis[1]))
        print("Jacobian foot: ", jacobian_lfoot)

        stacked_matrix = np.vstack((jacobian_lfoot, jacobian_rfoot))
        stacked_matrix = np.vstack((stacked_matrix, jacobian_pelvis))
        decomposed_matrix, _, _ = scipy.linalg.lu(stacked_matrix)
        print("Decomposed: ", decomposed_matrix)
        null_space = scipy.linalg.null_space(decomposed_matrix)
        print("null_space 1: ", null_space.shape)

        if action_vector is None:
            action_vector = np.random.uniform(-1, 1, len(q))
        action_vector = null_space*action_vector

        # normalise vector
        norm = np.linalg.norm(action_vector, ord=1)

        assert norm > 1e-6

        action_vector /= norm

        action_vector = np.multiply(action_vector, self.joint_increment)
        return action_vector


def rotX(theta):
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta), math.cos(theta)]])
    return R


def rotY(theta):
    R = np.array([
        [math.cos(theta), 0.0, math.sin(theta)],
        [0.0, 1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)]])
    return R


def rotZ(theta):
    R = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0]])
    return R


def quat_to_rot(qs):  # transform quaternion into rotation matrix
    qx = qs[0]
    qy = qs[1]
    qz = qs[2]
    qw = qs[3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty([3, 3])
    m[0, 0] = 1.0 - (yy + zz)
    m[0, 1] = xy - wz
    m[0, 2] = xz + wy
    m[1, 0] = xy + wz
    m[1, 1] = 1.0 - (xx + zz)
    m[1, 2] = yz - wx
    m[2, 0] = xz - wy
    m[2, 1] = yz + wx
    m[2, 2] = 1.0 - (xx + yy)

    return m


def euler_to_quat(roll, pitch, yaw):  # rad
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp

    return [x, y, z, w]


def rescale(value, old_range, new_range):
    value = np.array(value)
    old_range = np.array(old_range)
    new_range = np.array(new_range)

    OldRange = old_range[1][:] - old_range[0][:]
    NewRange = new_range[1][:] - new_range[0][:]
    NewValue = (value - old_range[0][:]) * \
        NewRange / OldRange + new_range[0][:]
    return NewValue

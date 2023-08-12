import numpy as np
import pickle
import math
import copy
import numpy
import matplotlib.pyplot as plt
import scipy
from scipy import signal

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class Motion():
    def __init__(self, dsr_data_freq=25, controlled_joints=None, dsr_gait_freq=0.7, stand_config=None):
        self.stand_config = stand_config
        self.dsr_data_freq = dsr_data_freq
        # list of name of joints controlled by agent
        self.controlled_joint_list = list(controlled_joints)
        self.available_joint_list = list([
            "torsoYaw", "torsoPitch", "torsoRoll",
            "rightHipYaw", "rightHipRoll", "rightHipPitch",
            "rightKneePitch",
            "rightAnklePitch", "rightAnkleRoll",
            "leftHipYaw", "leftHipRoll", "leftHipPitch",
            "leftKneePitch",
            "leftAnklePitch", "leftAnkleRoll", ])  # list of lower body joints that are available form human demonstration

        # search for controlled joints that are available in human reference
        self.reference_joint_list = []
        for key in self.available_joint_list:
            if key in self.controlled_joint_list:
                self.reference_joint_list.append(key)

        self.dsr_gait_freq = dsr_gait_freq
        # length of array for single gait cycle
        self.dsr_length = int(self.dsr_data_freq/self.dsr_gait_freq)

        self.motion_data_list = []
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s1ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s1ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s1ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s2ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s2ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s2ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s3ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s3ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s3ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s4ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s4ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s4ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s5ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s5ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s5ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s6ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s6ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s6ik3_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s7ik1_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s7ik2_processed.obj'))
        self.motion_data_list.append(self.load_mot(
            parentdir+'/human_demonstration/ik_files/s7ik3_processed.obj'))

        self.src_joint_angle_list = []
        self.src_joint_vel_list = []
        for motion_data in self.motion_data_list:
            joint_angle = self.getJointAngle(motion_data)
            joint_vel = self.getJointVel(joint_angle, motion_data)
            self.src_joint_angle_list.append(joint_angle)
            self.src_joint_vel_list.append(joint_vel)

        self.dsr_joint_angle_list = []
        self.dsr_joint_vel_list = []

        for i in range(len(self.motion_data_list)):
            joint_angle = self.src_joint_angle_list[i]
            joint_vel = self.src_joint_vel_list[i]
            motion_data = self.motion_data_list[i]

            dsr_joint_angle, dsr_joint_vel = self.process_data(
                joint_angle, joint_vel, motion_data)
            self.dsr_joint_angle_list.append(dsr_joint_angle)
            self.dsr_joint_vel_list.append(dsr_joint_vel)

        self.count = 0
        self.start = 0
        self.index = 1  # default is walking. 0 is kicking

        """Loading eef information"""  # this is using index 0 and assumes 25Hz

        self.eef_lfoot_pos = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_lfoot_pos.csv")
        self.eef_rfoot_pos = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_rfoot_pos.csv")
        self.eef_lfoot_contact = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_lfoot_contact.csv")
        self.eef_rfoot_contact = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_rfoot_contact.csv")
        self.eef_lfoot_orientation = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_lfoot_orientation.csv")
        self.eef_rfoot_orientation = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_rfoot_orientation.csv")
        self.eef_index = np.genfromtxt(
            parentdir+"/human_demonstration/ik_files/eef_imit/imit_index.csv")

    def getJointRange(self, margin):
        ref_joint_angles = self.dsr_joint_angle_list[self.index]

        joint_limit_lower = {}
        joint_limit_upper = {}
        for jointName in self.available_joint_list:
            joint_limit_lower.update(
                {jointName: min(ref_joint_angles[jointName])*3.14/180.-margin})
            joint_limit_upper.update(
                {jointName: max(ref_joint_angles[jointName])*3.14/180.+margin})
        return joint_limit_lower, joint_limit_upper

    def set_ref_to_stand_config(self):
        self.stand_config

    def load_mot(self, file_name='ik_files/s6ik1_processed.obj'):
        var = {}
        pkl_file = open(file_name + '', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def getJointAngle(self, motion_data):
        joint_angle = dict()
        motion_joint_data = motion_data['motion_data']

        surrogateTorsoYaw = np.zeros(
            np.shape(motion_joint_data['pelvis_rotation']))
        joint_angle.update({'torsoYaw': surrogateTorsoYaw.astype(float)})
        joint_angle.update(
            {'torsoPitch': -np.array(motion_joint_data['pelvis_tilt']).astype(float)})
        surrogateTorsoRoll = np.zeros(
            np.shape(motion_joint_data['pelvis_list']))
        joint_angle.update({'torsoRoll': surrogateTorsoRoll.astype(float)})

        surrogateRightHipYaw = np.zeros(
            np.shape(motion_joint_data['hip_rotation_r']))
        joint_angle.update({'rightHipYaw': surrogateRightHipYaw.astype(float)})
        joint_angle.update(
            {'rightHipRoll': -np.array(motion_joint_data['hip_adduction_r']).astype(float)})
        joint_angle.update(
            {'rightHipPitch': -np.array(motion_joint_data['hip_flexion_r']).astype(float)})
        joint_angle.update(
            {'rightKneePitch': -np.array(motion_joint_data['knee_angle_r']).astype(float)})
        joint_angle.update(
            {'rightAnklePitch': -np.array(motion_joint_data['ankle_angle_r']).astype(float)})
        joint_angle.update(
            {'rightAnkleRoll': -np.array(motion_joint_data['subtalar_angle_r']).astype(float)})

        surrogateLeftHipYaw = np.zeros(
            np.shape(motion_joint_data['hip_rotation_l']))
        joint_angle.update({'leftHipYaw': surrogateLeftHipYaw.astype(float)})
        joint_angle.update(
            {'leftHipRoll': -np.array(motion_joint_data['hip_adduction_l']).astype(float)})
        joint_angle.update(
            {'leftHipPitch': -np.array(motion_joint_data['hip_flexion_l']).astype(float)})
        joint_angle.update(
            {'leftKneePitch': -np.array(motion_joint_data['knee_angle_l']).astype(float)})
        joint_angle.update(
            {'leftAnklePitch': -np.array(motion_joint_data['ankle_angle_l']).astype(float)})
        joint_angle.update(
            {'leftAnkleRoll': -np.array(motion_joint_data['subtalar_angle_l']).astype(float)})
        return copy.deepcopy(joint_angle)

    def getJointVel(self, joint_angle, motion_data):

        joint_velocity = dict()
        for key, value in joint_angle.items():
            value = joint_angle[key]
            temp = []
            length = len(value)
            temp.extend(value)
            temp.extend(value)
            temp.extend(value)
            vel = []
            temp = np.array(temp)
            temp = temp.astype(float)
            for i in range(length, 2 * length):
                v = (temp[i + 1] - temp[i]) * motion_data['data_freq']
                vel.append(v)
            vel[-1] = (vel[-2]+vel[0])/2.0
            joint_velocity.update({key: vel})

        filter_joint_vel = dict()
        for key, value in joint_velocity.items():
            temp = []
            length = len(value)
            temp.extend(value)
            temp.extend(value)
            temp.extend(value)
            b, a = signal.butter(1, 0.25)
            y = signal.filtfilt(b, a, temp)
            vel = y[length:2 * length]
            filter_joint_vel.update({key: vel})

        return copy.deepcopy(filter_joint_vel)

    def process_data(self, joint_angle, joint_vel, motion_data):
        src_data_freq = motion_data['data_freq']
        src_gait_freq = motion_data['gait_freq']

        dsr_joint_angle = dict()
        for key, value in joint_angle.items():
            array = np.zeros(self.dsr_length)
            src_length = len(motion_data['motion_data']['time'])
            for i in range(self.dsr_length):
                index = min((i*src_length//self.dsr_length), src_length)
                array[i] = value[index]
            dsr_joint_angle.update({key: array})

        dsr_joint_vel = dict()
        for key, value in joint_vel.items():
            array = np.zeros(self.dsr_length)
            src_length = len(motion_data['motion_data']['time'])
            for i in range(self.dsr_length):
                index = min((i*src_length//self.dsr_length), src_length)
                # scale velocity using gait cycle frequency
                array[i] = value[index] * self.dsr_gait_freq / src_gait_freq
            dsr_joint_vel.update({key: array})

        return copy.deepcopy(dsr_joint_angle), copy.deepcopy(dsr_joint_vel)

    def ref_joint_angle(self):
        joint_angle = self.dsr_joint_angle_list[self.index]
        joint = []
        for i in range(len(self.reference_joint_list)):
            key = self.reference_joint_list[i]
            angle = joint_angle[key][self.count]*math.pi/180.0
            joint.append(angle)

            assert len(self.eef_lfoot_contact) == len(joint_angle[key]), "length of eef_lfoot_contact: %d, length of joint_angle[key]: %d" % (
                len(self.eef_lfoot_contact), len(joint_angle[key]))
        return joint

    def get_eef_data(self):
        eef_dict = {
            "eef_lfoot_pos": self.eef_lfoot_pos[self.count],
            "eef_rfoot_pos": self.eef_rfoot_pos[self.count],
            "eef_lfoot_contact": self.eef_lfoot_contact[self.count],
            "eef_rfoot_contact": self.eef_rfoot_contact[self.count],
            "eef_lfoot_orientation": self.eef_lfoot_orientation[self.count],
            "eef_rfoot_orientation": self.eef_rfoot_orientation[self.count],
        }
        return eef_dict

    def ref_joint_vel(self):
        joint_vel = self.dsr_joint_vel_list[self.index]
        joint = []
        for i in range(len(self.reference_joint_list)):
            key = self.reference_joint_list[i]
            vel = joint_vel[key][self.count]  # *math.pi/180.0
            joint.append(vel)
        joint = np.array(joint)*math.pi/180.0
        return joint

    def ref_joint_dict(self):
        joint_angle = self.dsr_joint_angle_list[self.index]
        joint_vel = self.dsr_joint_vel_list[self.index]
        joint_angle_array = []
        joint_vel_array = []
        for i in range(len(self.reference_joint_list)):
            key = self.reference_joint_list[i]
            angle = joint_angle[key][self.count]*math.pi/180.0
            vel = joint_vel[key][self.count]*math.pi/180.0
            joint_angle_array.append(angle)
            joint_vel_array.append(vel)

        return joint_angle_array, joint_vel_array

    def ref_joint_angle_dict(self, request_stand_config=False):
        if request_stand_config:
            joint_angles = self.stand_config
            joint = dict()
            for key in self.reference_joint_list:
                angle = joint_angles[key]
                joint.update({key: (angle)})
        else:
            joint_angles = self.dsr_joint_angle_list[self.index]
            joint = dict()
            for i in range(len(self.reference_joint_list)):
                key = self.reference_joint_list[i]
                angle = joint_angles[key][self.count]*math.pi/180.0
                joint.update({key: (angle)})

        return joint

    def ref_joint_vel_dict(self, request_stand_config=False):
        if request_stand_config:
            joint = dict()
            for key in self.reference_joint_list:
                joint.update({key: (0.0)})
        else:
            joint_vel = self.dsr_joint_vel_list[self.index]
            joint = dict()
            for i in range(len(self.reference_joint_list)):
                key = self.reference_joint_list[i]
                vel = joint_vel[key][self.count]*math.pi/180.0
                joint.update({key: (vel)})

        return joint

    def ref_motion_avg(self):
        joint = []
        for key in self.reference_joint_list:
            temp = 0
            length = len(self.dsr_joint_angle_list)
            for i in range(length):
                temp += self.dsr_joint_angle_list[i][key][self.count]
            temp /= length
            angle = temp*math.pi/180.0
            joint.append(angle)
        return joint

    def reset(self, index=None):
        if index is None:
            self.index = numpy.random.randint(len(self.dsr_joint_angle_list))
        else:
            self.index = index
        self.count = 0
        self.start = 0

    def index_count(self):
        self.count += 1
        self.count = self.count % self.dsr_length

    def random_count(self):
        self.count = numpy.random.randint(self.dsr_length)

    def set_count(self, count):
        self.count = count

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

    def get_base_orn(self):  # calculate base orientation from hip flexation
        left_max = max(self.dsr_joint_angle_list[self.index]['leftHipPitch'])
        left_min = min(self.dsr_joint_angle_list[self.index]['leftHipPitch'])
        right_max = max(self.dsr_joint_angle_list[self.index]['rightHipPitch'])
        right_min = min(self.dsr_joint_angle_list[self.index]['rightHipPitch'])

        offset_angle = -(left_max+right_max+left_min+right_min)/4.0 + 2.0
        offset_rad = offset_angle*math.pi/180.0

        quat = self.euler_to_quat(0.0, offset_rad, 0.0)
        return quat

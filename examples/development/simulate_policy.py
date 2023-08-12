import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts
from examples.development.loadProgress import load_progress

import numpy as np
import matplotlib.pyplot as plt

log_data = True

render = False

show_progress = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=500)
    parser.add_argument('--num-rollouts', '-n', type=int, default=1)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default='human',
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args


def simulate_policy(args):

    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')

    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    """Load training curves"""
    checkpoint_increment = variant["run_params"]["checkpoint_frequency"]
    best_checkpoint_return, best_checkpoint_idx = load_progress(
        checkpoint_path, show_progress=show_progress, checkpoint_increment=checkpoint_increment)
    print("==============================================================================")
    print("Best return: %d at checkpoint %d" %
          (best_checkpoint_return, best_checkpoint_idx))
    print("==============================================================================")

    checkpoint_path = experiment_path + "/checkpoint_%d" % best_checkpoint_idx

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    environment_params["kwargs"].update(
        {"renders": render, "visualise_goal": True, "save_trajectories": log_data})

    evaluation_environment = get_environment_from_params(environment_params)

    evaluation_environment._env.env._renders = render
    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    assert args.deterministic, "Polciy needs to be deterministic"

    args.render_mode = "human"
    with policy.set_deterministic(args.deterministic):
        paths = rollouts(args.num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=args.max_path_length,
                         render_mode=args.render_mode)

    """ Visualise Val trajecotries"""
    env_time = evaluation_environment._env.env.time_array
    q_imit_traj = evaluation_environment._env.env.q_imit_traj
    com_traj = np.array(evaluation_environment._env.env.com_traj)
    foot_force = evaluation_environment._env.env.foot_force

    rightHipRoll = []
    rightHipPitch = []
    rightKneePitch = []
    rightAnklePitch = []
    leftHipRoll = []
    leftHipPitch = []
    leftKneePitch = []
    leftAnklePitch = []

    for _val in q_imit_traj:
        rightHipRoll.append(_val["rightHipRoll"])
        rightHipPitch.append(_val["rightHipPitch"])
        rightKneePitch.append(_val["rightKneePitch"])
        rightAnklePitch.append(_val["rightAnklePitch"])
        leftHipRoll.append(_val["leftHipRoll"])
        leftHipPitch.append(_val["leftHipPitch"])
        leftKneePitch.append(_val["leftKneePitch"])
        leftAnklePitch.append(_val["leftAnklePitch"])

    rightHipRoll = np.array(rightHipRoll)
    rightHipPitch = np.array(rightHipPitch)
    rightKneePitch = np.array(rightKneePitch)
    rightAnklePitch = np.array(rightAnklePitch)
    leftHipRoll = np.array(leftHipRoll)
    leftHipPitch = np.array(leftHipPitch)
    leftKneePitch = np.array(leftKneePitch)
    leftAnklePitch = np.array(leftAnklePitch)

    eef_imit_traj = evaluation_environment._env.env.eef_imit_traj

    eef_lfoot_pos = []
    eef_rfoot_pos = []
    eef_lfoot_contact = []
    eef_rfoot_contact = []
    eef_lfoot_orientation = []
    eef_rfoot_orientation = []

    for _val in eef_imit_traj:
        eef_lfoot_pos.append(_val["eef_lfoot_pos"])
        eef_rfoot_pos.append(_val["eef_rfoot_pos"])
        eef_lfoot_contact.append(_val["eef_lfoot_contact"])
        eef_rfoot_contact.append(_val["eef_rfoot_contact"])
        eef_lfoot_orientation.append(_val["eef_lfoot_orientation"])
        eef_rfoot_orientation.append(_val["eef_rfoot_orientation"])
    eef_lfoot_pos = np.array(eef_lfoot_pos)
    eef_rfoot_pos = np.array(eef_rfoot_pos)
    eef_lfoot_contact = np.array(eef_lfoot_contact)
    eef_rfoot_contact = np.array(eef_rfoot_contact)
    eef_lfoot_orientation = np.array(eef_lfoot_orientation)
    eef_rfoot_orientation = np.array(eef_rfoot_orientation)
    q_real_traj = np.array(evaluation_environment._env.env.q_real_traj)
    action_traj = np.array(evaluation_environment._env.env.action_traj)
    eef_pose_traj = np.array(evaluation_environment._env.env.eef_pose_traj)
    vel_traj = np.array(evaluation_environment._env.env.vel_traj)
    eef_contact_traj = np.array(
        evaluation_environment._env.env.eef_contact_traj)

    pelvis_pos_traj = np.array(evaluation_environment._env.env.pelvis_pos_traj)
    pelvis_goal_traj = np.array(
        evaluation_environment._env.env.pelvis_goal_traj)
    vel_goal_traj = np.array(evaluation_environment._env.env.vel_goal_traj)
    gravity_traj = np.array(evaluation_environment._env.env.grav_traj)
    gravity_goal_traj = np.array(
        evaluation_environment._env.env.gravity_goal_traj)

    left_foot = np.array(
        [eef_pose_traj[:, 0], eef_pose_traj[:, 1], eef_pose_traj[:, 2]])
    right_foot = np.array(
        [eef_pose_traj[:, 4], eef_pose_traj[:, 5], eef_pose_traj[:, 6]])

    np.savetxt("lfoot.csv", left_foot, delimiter=",")
    np.savetxt("rfoot.csv", right_foot, delimiter=",")
    np.savetxt("com_pos.csv", com_traj, delimiter=",")
    pickle.dump(foot_force, open("foot_force.p", "wb"))
    if log_data:

        assert q_real_traj.shape[1] == 8

        fig = plt.figure(figsize=(18, 12))
        plt.subplot(3, 2, 1)
        plt.plot(env_time, 180/3.14*rightHipPitch,    label="Right imit (ref)")
        plt.plot(env_time, 180/3.14 *
                 q_real_traj[:, 1], label="Right real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 1], label="Right action right")
        plt.title("Right Hip Pitch")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(env_time, 180/3.14*leftHipPitch,     label="Left imit (ref)")
        plt.plot(env_time, 180/3.14*q_real_traj[:, 5], label="Left real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 5], label="Left action right")
        plt.legend()
        plt.title("Left Hip Pitch")

        plt.subplot(3, 2, 3)
        plt.plot(env_time, 180/3.14*rightKneePitch,   label="Right imit (ref)")
        plt.plot(env_time, 180/3.14 *
                 q_real_traj[:, 2], label="Right Real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 2], label="Right Action right")
        plt.title("Right Knee Pitch")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(env_time, 180/3.14*leftKneePitch,    label="Left imit (ref)")
        plt.plot(env_time, 180/3.14*q_real_traj[:, 6], label="Left Real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 6], label="Left Action right")
        plt.legend()
        plt.title("Left Knee Pitch")

        plt.subplot(3, 2, 5)
        plt.plot(env_time, 180/3.14*rightAnklePitch,  label="Right imit (ref)")
        plt.plot(env_time, 180/3.14 *
                 q_real_traj[:, 3], label="Right Real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 3], label="Right Action right")
        plt.title("Right Ankle Pitch")
        plt.legend()

        plt.subplot(3, 2, 6)
        plt.plot(env_time, 180/3.14*leftAnklePitch,   label="Left imit (ref)")
        plt.plot(env_time, 180/3.14*q_real_traj[:, 7], label="Left Real right")
        plt.plot(env_time, 180/3.14 *
                 action_traj[:, 7], label="Left Action right")
        plt.legend()
        plt.title("Left Ankle Pitch")

        fig = plt.figure(figsize=(18, 12))
        plt.subplot(2, 2, 1)
        plt.plot(env_time, eef_rfoot_pos[:, 0], label="Right imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 4], label="Right real")
        plt.legend()
        plt.title("Right Foot eef x")

        plt.subplot(2, 2, 2)
        plt.plot(env_time, eef_lfoot_pos[:, 0], label="Left imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 0], label="Left real")
        plt.title("Left Foot eef x")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(env_time, eef_rfoot_pos[:, 1], label="Right imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 5], label="Right real")
        plt.legend()
        plt.title("Right Foot eef y")

        plt.subplot(2, 2, 4)
        plt.plot(env_time, eef_lfoot_pos[:, 1], label="Left imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 1], label="Left real")
        plt.title("Left Foot eef y")
        plt.legend()

        fig = plt.figure(figsize=(18, 12))
        z_left_offset = min(min(eef_lfoot_pos[:, 2]), min(eef_pose_traj[:, 2]))
        z_right_offset = min(
            min(eef_rfoot_pos[:, 2]), min(eef_pose_traj[:, 6]))
        plt.plot()
        plt.subplot(3, 1, 1)
        plt.plot(env_time, eef_rfoot_pos[:, 2] -
                 z_right_offset, label="Right imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 6] -
                 z_right_offset, label="Right real")
        plt.plot(env_time, eef_lfoot_pos[:, 2] -
                 z_left_offset, label="Left imit (ref)")
        plt.plot(env_time, eef_pose_traj[:, 2] -
                 z_left_offset, label="Left real")
        plt.title("Foot eef z")
        plt.legend()

        plt.plot()
        plt.subplot(3, 1, 2)
        plt.plot(env_time, 180/3.14 *
                 eef_rfoot_orientation[:, 1], label="Right imit (ref)")
        plt.plot(env_time, 180/3.14*eef_pose_traj[:, 7], label="Right real")
        plt.plot(env_time, 180/3.14 *
                 eef_lfoot_orientation[:, 1], label="Left imit (ref)")
        plt.plot(env_time, 180/3.14*eef_pose_traj[:, 3], label="Left real")
        plt.legend()
        plt.title("Foot pitch")

        plt.subplot(3, 1, 3)
        plt.scatter(env_time, eef_lfoot_contact*1.0, label="Left imit (ref)")
        plt.scatter(env_time, eef_contact_traj[:, 0]*1.1, label="Left real")
        plt.scatter(env_time, eef_rfoot_contact*1.3, label="Right imit (ref)")
        plt.scatter(env_time, eef_contact_traj[:, 1]*1.4, label="Right real")
        plt.ylim([0.9, 1.5])
        plt.title("Foot eef contact")
        plt.legend()
        window_length = 3
        velx_average = moving_average(vel_traj[:, 0], n=window_length)
        vely_average = moving_average(vel_traj[:, 1], n=window_length)

        fig = plt.figure(figsize=(18, 12))
        plt.subplot(3, 1, 1)
        plt.plot(env_time, vel_traj[:, 0], label="X vel real")
        plt.plot(env_time[:-(window_length-1)], velx_average,
                 label="X vel moving average real")
        plt.plot(env_time, vel_goal_traj[:, 0], label="X vel ref")
        plt.title("X vel over time")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(env_time, vel_traj[:, 1], label="Y vel real")
        plt.plot(env_time[:-(window_length-1)], vely_average,
                 label="Y vel moving average real")
        plt.plot(env_time, vel_goal_traj[:, 1], label="Y vel ref")
        plt.title("Y vel over time")
        plt.legend()

        plt.subplot(3, 1, 3)

        for idx, val in enumerate(velx_average):
            plt.plot([0, velx_average[idx]], [
                     0, vely_average[idx]], label="vel real", c="k")
        plt.plot([0., vel_goal_traj[-1, 0]],
                 [0., vel_goal_traj[-1, 1]], label="Vel ref", c="r")
        plt.xlim([-0.1, 1])
        plt.ylim([-1, 1])
        plt.title("X and y vel in cartesian space")

        fig = plt.figure(figsize=(18, 12))
        plt.subplot(2, 1, 1)
        plt.plot(env_time, pelvis_pos_traj[:, 0], label="X pos real")
        plt.plot(env_time, pelvis_pos_traj[:, 1], label="Y pos real")
        plt.plot(env_time, pelvis_goal_traj[:, 0], label="X pos ref")
        plt.plot(env_time, pelvis_goal_traj[:, 1], label="Y pos ref")
        plt.title("X and y pos over time")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(pelvis_pos_traj[:, 0],
                 pelvis_pos_traj[:, 1], label="Pos real")
        plt.scatter(pelvis_goal_traj[:, 0],
                    pelvis_goal_traj[:, 1], label="Pos ref", s=4)
        plt.title("X and y pos in cartesian space")
        plt.xlim([-0.1, max(max(pelvis_goal_traj[:, 0]),
                            max(pelvis_pos_traj[:, 0]))+0.1])
        plt.ylim([min(min(pelvis_goal_traj[:, 1]), min(pelvis_pos_traj[:, 1])) -
                  0.1, max(max(pelvis_goal_traj[:, 1]), max(pelvis_pos_traj[:, 1]))+0.1])
        plt.legend()

        fig = plt.figure(figsize=(18, 12))
        plt.subplot(2, 1, 1)
        plt.plot(env_time, gravity_traj[:, 0], label="X orientation real")
        plt.plot(env_time, gravity_traj[:, 1], label="Y orientation real")
        plt.plot(env_time, gravity_goal_traj[:, 0], label="X orientation ref")
        plt.plot(env_time, gravity_goal_traj[:, 1], label="Y orientation ref")
        plt.title("X and y orientation over time")
        plt.legend()

        plt.subplot(2, 1, 2)
        for gravity_val in gravity_traj:
            plt.plot([0, gravity_val[0]], [0, gravity_val[1]],
                     label="orientation real", c="k")
        plt.plot([0, gravity_goal_traj[-1, 0]], [0, gravity_goal_traj[-1, 1]],
                 label="orientation ref", linewidth=2.0, c="r")
        plt.title("X and y orientation in cartesian space")
        plt.xlim([-0.1, 1])
        plt.ylim([min(min(gravity_goal_traj[:, 1]), min(gravity_traj[:, 1])) -
                  0.1, max(max(gravity_goal_traj[:, 1]), max(gravity_traj[:, 1]))+0.1])

        plt.show()

    return paths


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    args = parse_args()
    simulate_policy(args)

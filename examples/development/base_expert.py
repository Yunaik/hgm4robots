from ray import tune
import numpy as np
import pdb

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 2000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Pendulum': 200,
    'Valkyrie': 500
}

ALGORITHM_PARAMS_ADDITIONAL = {
    'MBPO': {
        'type': 'MBPO',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        },
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    },
    'MVE': {
        'type': 'MVE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
        }
    },
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(10000),
        },
    },
}

DEFAULT_NUM_EPOCHS = 2000

NUM_EPOCHS_PER_DOMAIN = {
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pendulum': 10,
    "Valkyrie": 6000
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}

ENVIRONMENT_PARAMS = {
    "Valkyrie":
    {
        "v0": {
            "terminate_if_not_double_support": False,
            "filter_action": False,
            "useFullDOF": False,
            "incremental_control": True,
            "margin_in_degree": 20,
            "time_to_stabilise": 2.0,
            "goal_type": "random_fixed",  # fixed, random_fixed, moving_goal
        },

        "v1": {  # reach
            "useCollision": False,
            "incremental_control": True,
            "margin_in_degree": 1000,
            "time_to_stabilise": 2.0,
            "clamp_object": False,
            "calculate_PD_from_torque": True,
            "random_joint_init": False,
            "goal_type": "fixed",  # fixed, random_fixed, moving_goal
            "weight_dic": {
                "weight_lx_pos_reward":         1.0/3.0,
                "weight_ly_pos_reward":         1.0/3.0,
                "weight_lz_pos_reward":         1.0/3.0,
                "weight_rx_pos_reward":         2.0/3.0,
                "weight_ry_pos_reward":         2.0/3.0,
                "weight_rz_pos_reward":         3.0/3.0,
                "weight_lx_vel_reward":         1.0/6.0,
                "weight_ly_vel_reward":         1.0/6.0,
                "weight_lz_vel_reward":         1.0/6.0,
                "weight_rx_vel_reward":         1.0/6.0,
                "weight_ry_vel_reward":         1.0/6.0,
                "weight_rz_vel_reward":         1.0/6.0,
                "weight_joint_vel_reward":      1.0/4.0,
                "weight_joint_torque_reward":   1.0/4.0,

                # Clamp reward
                "weight_box_pos_x_reward":      1.0/3.0,
                "weight_box_pos_y_reward":      1.0/3.0,
                "weight_box_pos_z_reward":      1.0/3.0,
                "weight_contact_reward":        1.0,
                "weight_box_gravity_reward":    0.5,
                "weight_box_vel_x_reward":      0.0/6.0,
                "weight_box_vel_y_reward":      0.0/6.0,
                "weight_box_vel_z_reward":      0.0/6.0,
            },
        },

        "v2": {  # forward locomotion
            # Fixed
            "time_to_stabilise": 0.0,
            "imitate_motion": True,
            "lock_upper_body": True,
            "target_velocity": [0.5, 0, 0],
            "goal_type": None,  # fixed, random_fixed, moving_goal, None

            # To change
            "learn_stand": False,

            "filter_action": False,
            "action_bandwidth": 12,

            "imit_weights": {"imitation": 0.8, "goal": 0.2},

            "joint_imit_tolerance": {'torsoPitch':  22.5,
                                     'rightHipRoll': 22.5, 'rightHipPitch': 22.5, 'rightKneePitch': 22.5, 'rightAnklePitch': 22.5, 'rightAnkleRoll': 22.5,
                                     'leftHipRoll':  22.5, 'leftHipPitch':  22.5, 'leftKneePitch':  22.5, 'leftAnklePitch':  22.5, 'leftAnkleRoll':  22.5},

            "weight_dic": {
                "weight_x_pos_reward":            0.0,
                "weight_y_pos_reward":            0.0,
                "weight_torso_pitch_reward":      0.0,
                "weight_pelvis_pitch_reward":     0.0,
                "weight_left_foot_force_reward":  0.0,
                "weight_right_foot_force_reward": 0.0,
                "weight_foot_clearance_reward":   0.0,
                "weight_foot_pitch_reward":       0.0,

                "weight_x_vel_reward":            8.0,
                "weight_y_vel_reward":            1.0,
                "weight_z_vel_reward":            1.0,
                "weight_z_pos_reward":            1.0,
                "weight_gravity_reward":          1.0,
                "weight_joint_vel_reward":        0.5,
                "weight_joint_torque_reward":     0.5,
                "weight_foot_contact_reward":     1.0,
                "weight_foot_slippage_reward":    0.0,

                "imit_joint_pos_reward":          0.5,
                "imit_eef_contact_reward":        0.2,
                "imit_eef_pos_reward":            0.2,
                "imit_eef_orientation_reward":    0.1,
            },

            "joint_weights": {
                "rightHipRoll":     1,
                "rightHipPitch":    4,
                "rightKneePitch":   4,
                "rightAnklePitch":  2,
                "leftHipRoll":      1,
                "leftHipPitch":     4,
                "leftKneePitch":    4,
                "leftAnklePitch":   2,
            }
        },

        "v3": {  # standing
            # Fixed
            "time_to_stabilise": 0.0,
            "imitate_motion": True,
            "lock_upper_body": True,
            "target_velocity": [0, 0, 0],
            "goal_type": None,  # fixed, random_fixed, moving_goal, None

            # To change
            "learn_stand": True,
            # "require_full_contact_foot": True,
            "require_full_contact_foot": False,
            "exertForce": False,
            # "exertForce": True,
            "filter_action": False,
            "action_bandwidth": 12,

            "imit_weights": {"imitation": 0.5, "goal": 0.5},

            "joint_imit_tolerance": {'torsoPitch':  22.5,
                                     'rightHipRoll': 22.5, 'rightHipPitch': 22.5, 'rightKneePitch': 22.5, 'rightAnklePitch': 22.5, 'rightAnkleRoll': 22.5,
                                     'leftHipRoll':  22.5, 'leftHipPitch':  22.5, 'leftKneePitch':  22.5, 'leftAnklePitch':  22.5, 'leftAnkleRoll':  22.5},

            "weight_dic": {
                "weight_x_pos_reward":            0.0,
                "weight_y_pos_reward":            0.0,
                "weight_torso_pitch_reward":      0.0,
                "weight_pelvis_pitch_reward":     0.0,
                "weight_left_foot_force_reward":  0.0,
                "weight_right_foot_force_reward": 0.0,
                "weight_foot_clearance_reward":   0.0,
                "weight_foot_pitch_reward":       0.0,

                "weight_x_vel_reward":            2.0,
                "weight_y_vel_reward":            2.0,
                "weight_z_vel_reward":            2.0,
                "weight_z_pos_reward":            6.0,
                "weight_gravity_reward":          1.0,
                "weight_joint_vel_reward":        0.5,
                "weight_joint_torque_reward":     0.5,
                "weight_foot_contact_reward":     2.0,
                "weight_foot_slippage_reward":    0.0,

                "imit_joint_pos_reward":          1.0,
                "imit_eef_contact_reward":        0.0,
                "imit_eef_pos_reward":            0.0,
                "imit_eef_orientation_reward":    0.0,
            },

            "joint_weights": {
                "rightHipRoll":     1,
                "rightHipPitch":    4,
                "rightKneePitch":   4,
                "rightAnklePitch":  2,
                "leftHipRoll":      1,
                "leftHipPitch":     4,
                "leftKneePitch":    4,
                "leftAnklePitch":   2,
            }
        },

        "v4": {  # follow goal
            # Fixed
            "terminate_if_pelvis_out_of_range": False,
            "time_to_stabilise": 0.0,
            "imitate_motion": True,
            "lock_upper_body": True,
            "target_velocity": [0.5, 0., 0],
            "goal_type": "random_fixed",  # fixed, random_fixed, moving_goal, None
            "goal_y_range": 0.0,
            "filter_action": False,
            "action_bandwidth": 12,
            "obs_use_yaw": True,
            "tighter_tolerance_upon_reaching_goal": False,
            # "obs_use_pos": True,
            "obs_use_pos": False,

            "goal_as_vel": False,
            "reach_short_distance": False,

            "imit_weights": {"imitation": 0.5, "goal": 0.5},

            "joint_imit_tolerance": {'torsoPitch':  22.5,
                                     'rightHipRoll': 22.5, 'rightHipPitch': 22.5, 'rightKneePitch': 22.5, 'rightAnklePitch': 22.5, 'rightAnkleRoll': 22.5,
                                     'leftHipRoll':  22.5, 'leftHipPitch':  22.5, 'leftKneePitch':  22.5, 'leftAnklePitch':  22.5, 'leftAnkleRoll':  22.5},

            "weight_dic": {
                "weight_torso_pitch_reward":      0.0,
                "weight_pelvis_pitch_reward":     0.0,
                "weight_left_foot_force_reward":  0.0,
                "weight_right_foot_force_reward": 0.0,
                "weight_foot_clearance_reward":   0.0,
                "weight_foot_pitch_reward":       0.0,

                "weight_x_vel_reward":            1.0,
                "weight_y_vel_reward":            1.0,
                "weight_z_vel_reward":            0.5,
                "weight_x_pos_reward":            1.5,
                "weight_y_pos_reward":            1.5,
                "weight_z_pos_reward":            0.5,
                "weight_gravity_reward":          2.0,
                "weight_joint_vel_reward":        0.0,
                "weight_joint_torque_reward":     0.0,
                "weight_foot_contact_reward":     1.0,
                "weight_foot_slippage_reward":    0.0,

                "imit_joint_pos_reward":          0.5,
                "imit_eef_contact_reward":        0.2,
                "imit_eef_pos_reward":            0.2,
                "imit_eef_orientation_reward":    0.1,
            },

            "joint_weights": {
                "rightHipRoll":     1,
                "rightHipPitch":    4,
                "rightKneePitch":   4,
                "rightAnklePitch":  2,
                "leftHipRoll":      1,
                "leftHipPitch":     4,
                "leftKneePitch":    4,
                "leftAnklePitch":   2,
            }
        },
    }
}

NUM_CHECKPOINTS = 12


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_params):

    algorithm_params = deep_update(
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {}),
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        env_params
    )

    variant_spec = {
        'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    return variant_spec


def get_variant_spec(args, env_params):
    universe, domain, task = env_params.universe, env_params.domain, env_params.task

    variant_spec = get_variant_spec_base(
        universe, domain, task, args.policy, env_params.type, env_params)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec

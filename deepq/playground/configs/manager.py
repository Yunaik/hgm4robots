import json
import importlib
import gym
import os
from gym.wrappers import Monitor
from deepq.playground.utils.misc import plot_from_monitor_results
import valkyrie_gym_env


def load_policy_class(policy_name):
    mod = importlib.import_module("deepq.playground.policies")
    policy_class = getattr(mod, policy_name)
    return policy_class


def load_wrapper_class(wrapper_name):
    mod = importlib.import_module("deepq.playground.utils.wrappers")
    wrapper_class = getattr(mod, wrapper_name)
    return wrapper_class


def apply_wrappers(env, list_of_wrappers):
    for name, params in list_of_wrappers:
        wrapper_class = load_wrapper_class(name)
        env = wrapper_class(env, **params)
    return env


class ConfigManager:
    def __init__(self, env_name, policy_name, policy_params=None, train_params=None,
                 wrappers=None):
        self.env_name = env_name
        self.policy_name = policy_name
        self.policy_params = policy_params or {}
        self.train_params = train_params or {}
        self.wrappers = wrappers or []

        self.env = gym.make("Valkyrie-v5",
                            incremental_control=False,
                            incremental_control_high=False,
                            renders=False,
                            time_to_stabilise=0.,
                            goal_type="random_fixed",
                            visualise_goal=True,
                            random_joint_init=False,
                            spawn_objects=True,
                            fixed_base=False,
                            goal_as_vel=False,
                            target_velocity=[0.5, 0., 0.],
                            control_mode="whole_body",
                            obs_use_yaw=True,
                            imitate_motion=True,
                            final_goal_type="right",
                            print_reward_details=False,
                            filter_action=True,
                            action_bandwidth=9,
                            load_obstacle=0,
                            save_trajectories=True,
                            high_level_frequency=0.5,
                            random_spawn=True,
                            reward_pose_penalty=1,
                            reward_box_in_hand=1,
                            reward_box_drop=2,
                            reward_door_is_open=2,
                            reward_is_at_goal=5,
                            )

        self.env = apply_wrappers(self.env, self.wrappers)

    def to_json(self):
        return dict(
            env_name=self.env_name,
            wrappers=self.wrappers,
            policy_name=self.policy_name,
            policy_params=self.policy_params,
            train_params=self.train_params,
        )

    @classmethod
    def load(cls, file_path):
        assert os.path.exists(file_path)
        return cls(**json.load(open(file_path)))

    def save(self, file_path):
        with open(file_path, 'w') as fin:
            json.dump(self.to_json(), fin, indent=4, sort_keys=True)

    def load_policy(self, model_name):
        env = self.env
        self.policy = load_policy_class(self.policy_name)(
            env, model_name, training=True, **self.policy_params)
        self.policy.build()

    def start_training(self, model_name):
        self.env.reset()
        self.load_policy(model_name)
        print("\n==================================================")
        print("Loaded gym.env:", self.env_name)
        print("Wrappers:", self.wrappers)
        print("Loaded policy:", self.policy.__class__)
        print("Policy params:", self.policy_params)
        print("Train params:", self.train_params)
        print("==================================================\n")

        train_config = self.policy.TrainConfig(**self.train_params)
        self.policy.train(train_config)

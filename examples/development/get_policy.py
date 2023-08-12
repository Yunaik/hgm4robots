import json
import os
import pickle

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant


def getPolicy(checkpoint_path):
    session = tf.keras.backend.get_session()
    checkpoint_path = checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    evaluation_environment = get_environment_from_params(environment_params)

    evaluation_environment._env.env._renders = True
    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    policy.set_deterministic(True)
    return policy, evaluation_environment

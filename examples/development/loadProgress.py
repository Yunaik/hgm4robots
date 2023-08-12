import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import copy

checkpoint_path = "foo/bar/checkpoint_1500"


def load_progress(checkpoint_path, show_progress=True, checkpoint_increment=500):
    print("If the learning curves look weird, check the rolling average window size (may be too large depending on the size of epochs)")
    checkpoint_path = checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)
    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)
    reward_dict = variant["environment_params"]["evaluation"]["kwargs"]["weight_dic"]
    try:
        weight_imit = variant["environment_params"]["evaluation"]["kwargs"]["imit_weights"]["imitation"]
        weight_goal = variant["environment_params"]["evaluation"]["kwargs"]["imit_weights"]["goal"]
    except:
        weight_imit = None
        weight_goal = None
    _input = pd.read_csv(experiment_path + "/progress.csv", sep=",")
    epochs = _input["epoch"]
    reward_keys = {}

    for key in list(_input.columns):
        if "-mean-mean" in key:
            reward_keys.update({key: key.replace("-mean-mean", "")})

    """Get best training and test return"""
    best_training_return = -99999
    best_training_returns = []
    best_training_epochs = []
    for idx, _return in enumerate(_input["training/return-average"]):
        if _return > best_training_return:
            best_training_return = _return
            best_training_returns.append(_return)
            best_training_epochs.append(epochs[idx])

    best_training_epochs.append(epochs.iloc[-1])
    best_training_returns.append(best_training_returns[-1])

    best_test_return = -99999
    best_test_returns = []
    best_test_epochs = []
    best_checkpoint_idx = 0
    best_checkpoint_return = -999999
    for idx, _return in enumerate(_input["evaluation/return-average"]):
        if _return > best_test_return:
            best_test_return = _return
            best_test_returns.append(_return)
            best_test_epochs.append(epochs[idx])
        if idx % checkpoint_increment == 0:
            if _return > best_checkpoint_return:
                best_checkpoint_return = _return
                best_checkpoint_idx = idx if idx != 0 else checkpoint_increment

    best_test_epochs.append(epochs.iloc[-1])
    best_test_returns.append(best_test_returns[-1])

    """ Training return """
    rolling_average_window = 8
    if show_progress:

        fig = plt.figure(figsize=(18, 6))
        plt.subplot(2, 2, 1)
        plt.title('Training returns')
        plt.plot(epochs, _input["training/return-average"].rolling(
            window=rolling_average_window).mean(), label='Average return', alpha=1.0)
        plt.plot(epochs, _input["training/return-min"].rolling(
            window=rolling_average_window).mean(), label='Min return', alpha=0.3)
        plt.plot(epochs, _input["training/return-max"].rolling(
            window=rolling_average_window).mean(), label='Max Return', alpha=0.3)
        plt.fill_between(epochs, _input["training/return-average"].rolling(window=rolling_average_window).mean()+_input["training/return-std"].rolling(window=rolling_average_window).mean(),
                         _input["training/return-average"].rolling(window=rolling_average_window).mean()-_input["training/return-std"].rolling(window=rolling_average_window).mean(), facecolor='blue', alpha=0.1)
        plt.plot(best_training_epochs, best_training_returns,
                 label="Best training returns")
        plt.legend(loc='best', bbox_to_anchor=(1, 1))

        plt.subplot(2, 2, 3)
        plt.title("Training episode length")
        plt.plot(
            epochs, _input["training/episode-length-avg"].rolling(window=rolling_average_window).mean())
        """ Test return """
        rolling_average_window = 40

        plt.subplot(2, 2, 2)
        plt.title('Test returns')
        plt.plot(epochs, _input["evaluation/return-average"].rolling(
            window=rolling_average_window).mean(), label='Average return', alpha=1.0)
        plt.plot(epochs, _input["evaluation/return-min"].rolling(
            window=rolling_average_window).mean(), label='Min return', alpha=0.3)
        plt.plot(epochs, _input["evaluation/return-max"].rolling(
            window=rolling_average_window).mean(), label='Max Return', alpha=0.3)
        plt.fill_between(epochs, _input["evaluation/return-average"].rolling(window=rolling_average_window).mean()+_input["evaluation/return-std"].rolling(window=rolling_average_window).mean(),
                         _input["evaluation/return-average"].rolling(window=rolling_average_window).mean(
        )-_input["evaluation/return-std"].rolling(window=rolling_average_window).mean(),
            facecolor='blue', alpha=0.1)
        plt.plot(best_test_epochs, best_test_returns,
                 label="Best test returns")
        plt.legend(loc='best', bbox_to_anchor=(1, 1))

        plt.subplot(2, 2, 4)
        plt.title("Test episode length")
        plt.plot(epochs, _input["evaluation/episode-length-avg"].rolling(
            window=rolling_average_window).mean())
        """ Training return by sub rewards """
        rolling_average_window = 40

        if weight_imit:
            fig = plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            for key in reward_keys:
                if "evaluation" in key and "imitation" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")
                    if name == "imitation_contact_reward":
                        weight = reward_dict["imit_eef_contact_reward"]
                    elif name == "imitation_foot_orientation_reward":
                        weight = reward_dict["imit_eef_orientation_reward"]
                    elif name == "imitation_foot_pos_reward":
                        weight = reward_dict["imit_eef_pos_reward"]
                    elif name == "imitation_joint_pos_reward":
                        weight = reward_dict["imit_joint_pos_reward"]
                    elif name == "imitation_contact_term":
                        weight = reward_dict["imit_eef_contact_reward"]
                    elif name == "imitation_joint_vel_reward":
                        weight = 1
                    elif name == "reward_imitation":
                        weight = 10
                    else:
                        assert 3 == 4, "Term %s does not exist" % name

                    if name == "reward_imitation":
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean()/weight, label=name, linewidth=3)
                    else:
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean()/weight, label=name)
                    plt.legend()

        if weight_goal:
            plt.subplot(1, 2, 2)
            for key in reward_keys:
                if "evaluation" in key and not "imitation" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")
                    if name == "reward_goal":
                        weight = 10
                    else:
                        weight = reward_dict["weight_"+name]
                    if weight:
                        if name == "reward_goal":
                            plt.plot(epochs, _input[key].rolling(
                                window=rolling_average_window).mean()/weight, label=name, linewidth=3)
                        else:
                            plt.plot(epochs, _input[key].rolling(
                                window=rolling_average_window).mean()/weight, label=name)
                    plt.legend()

        if weight_goal is None and weight_imit is None:
            fig = plt.figure(figsize=(18, 6))

            plt.subplot(1, 2, 1)
            for key in reward_keys:
                if "evaluation" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")
                    if "l" in name[0] and "pos" in name and not "joint" in name:
                        weight = reward_dict["weight_"+name]
                        if weight:
                            plt.plot(epochs, _input[key].rolling(
                                window=rolling_average_window).mean(), label=name)
                        plt.legend()
            plt.subplot(1, 2, 2)
            for key in reward_keys:
                if "evaluation" in key and "r" in key and "pos" in key and not "joint" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")
                    if "r" in name[0] and "pos" in name and not "joint" in name:
                        weight = reward_dict["weight_"+name]
                        if weight:
                            plt.plot(epochs, _input[key].rolling(
                                window=rolling_average_window).mean(), label=name)
                        plt.legend()

            fig = plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            for key in reward_keys:
                if "evaluation" in key and "vel" in key and not "joint" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")

                    weight = reward_dict["weight_"+name]
                    if weight:
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean(), label=name)
                    plt.legend()
            plt.subplot(1, 2, 2)
            for key in reward_keys:
                if "evaluation" in key and "joint" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")

                    weight = reward_dict["weight_"+name]
                    if weight:
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean(), label=name)
                    plt.legend()

            fig = plt.figure(figsize=(18, 6))
            plt.subplot(1, 2, 1)
            for key in reward_keys:
                if "evaluation" in key and "vel" not in key and "joint" not in key and "pos" not in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")

                    weight = reward_dict["weight_"+name]
                    if weight:
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean(), label=name)
                    plt.legend()
            plt.subplot(1, 2, 2)
            for key in reward_keys:
                if "evaluation" in key and "box_pos" in key:
                    name = copy.copy(key)
                    name = name.replace("evaluation/env_infos/", "")
                    name = name.replace("-mean-mean", "")

                    weight = reward_dict["weight_"+name]
                    if weight:
                        plt.plot(epochs, _input[key].rolling(
                            window=rolling_average_window).mean(), label=name)
                    plt.legend()
        plt.show()
    return best_checkpoint_return, best_checkpoint_idx

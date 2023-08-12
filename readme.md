# Usage

This code is structured into three main parts: high-level policy training, mid-level policy training, and low-level.

As the proposed method is a hierarchical generative model, the lower levels need to be implemented first, before the higher levels can be trained. Thus, the low-level controller needs to be first tuned. Then, the mid-level policy can be trained, and lastly, the high-level policy is trained that uses both mid-level and low-level policy.

After successfully obtaining all three policies, the task as outlined in the paper can be achieved by running `qlearning/test_val.py`.

## Pre-trained policies and assets
Please download the policies from [this link](https://drive.google.com/drive/folders/1m-XaJdmOEyPfPZdxbCDi94paE82PIRxu?usp=sharing) and extract to `env/valkyrie/valkyrie_gym_env/envs/nn_policies`.

please download the urdf files from [this link](https://drive.google.com/drive/folders/1m-XaJdmOEyPfPZdxbCDi94paE82PIRxu?usp=sharing) and extract to `env/valkyrie/valkyrie_gym_env/envs/urdf`

## High-level policy
The high-level policy is trained using double Q-Learning, which is implemented in `qlearning/learn_val.py`. The policy is tested by running `qlearning/test_val.py`.

## Mid-level policy
The mid-level policy for locomotion is trained via the Soft Actor-Critic (SAC), which is implemented in the `mbpo` module, that concurrently uses functionalities from the `softlearning` module. The mid-level policy (minimum jerk control) for manipulation is directly implemented in `env/valkyrie/valkyrie_gym_env/envs/valkyrie.py`.

After installation of `mbpo` (see `mbpo_readme.md`), the locomotion policy is trained by running

```
mbpo run_local examples.development --config=examples.config.custom.valkyrie_walk
```

## Low-level policy
The low level functionality is implemented in the Valkyrie class in `env/valkyrie/valkyrie_gym_env/envs/valkyrie.py`. This class serves as environment for simulation as well.

# Acknowledgement

This repository uses code from the repositories [mbpo](https://github.com/JannerM/mbpo) for SAC and [deep-reinforcement-learning-gym](https://github.com/lilianweng/deep-reinforcement-learning-gym) for Double Q-Learning, which were developed by Michael Janner and [Lilian Weng](https://lilianweng.github.io/lil-log/) respectively.


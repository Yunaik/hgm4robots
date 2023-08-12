from collections import defaultdict

import numpy as np
import time
from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self.action_time = []
        self.step_time = []
        self.process_time = []
        self.image_list=[]
        self.save_video=False
    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
        action_time = time.time()
        # self._current_observation = np.array([-0.57531866,  0.81792936,  0.11435547,  2.38824338,  0.03207261,  0.10326705,
        #                             -0.20595599, -0.07169492 ,-0.03061513 ,-0.13390651, -0.99052095 , 0.26093243,
        #                             0.40881234,  0.15346006, -0.03377815 ,-0.36384325,  0.47365484 ,-0.12972442,
        #                             0.1089697,   0.06869599 , 0.13751951 ,-0.27588716])
        # print("Obs: ", self._current_observation)
        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]
        self.action_time.append(time.time()-action_time)
        step_time = time.time()
        # print("Action: ", action)
        next_observation, reward, terminal, info = self.env.step(action)
        if self.save_video:
            self.image_list.append(self.env.render(distance=7, yaw=45, pitch=0, roll=0,)) # standard

        self.step_time.append(time.time()-step_time)

        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        process_time = time.time()
        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)
        # print("Max path length: ", self._max_path_length)
        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation
        self.process_time.append(time.time()-process_time)
        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        ac = np.mean(np.array(self.action_time))
        st = np.mean(np.array(self.step_time))
        pr = np.mean(np.array(self.process_time))
        total_time = ac+st+pr
        print("Action: %.2fms (%.2f), step: %.2fms (%.2f), pr: %.2fms (%.2f). Total: %.2fms " 
        % (ac*1e3, ac/total_time, st*1e3, st/total_time, pr*1e3, pr/total_time, total_time*1e3))
        return diagnostics

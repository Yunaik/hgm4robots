import copy
import pickle
import time
from deepq.playground.utils.tf_ops import dense_nn, conv2d_net, lstm_net
from deepq.playground.utils.misc import plot_learning_curve
from deepq.playground.policies.memory import ReplayMemory, ReplayTrajMemory, Transition
from deepq.playground.policies.base import BaseModelMixin, Policy, TrainConfig
from gym.spaces import Box, Discrete
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DqnWarmstartPolicy(Policy, BaseModelMixin):
    def __init__(self, env, name,
                 training=True,
                 gamma=0.99,
                 batch_size=256,
                 model_type='dense',
                 model_params=None,
                 step_size=1,  # only > 1 if model_type is 'lstm'.
                 layer_sizes=[32, 32],
                 double_q=True,
                 dueling=False,
                 memory_capacity=int(1e6),
                 do_rollout=True):
        """
        model_params: 'layer_sizes', 'step_size', 'lstm_layers', 'lstm_size'
        """
        Policy.__init__(self, env, name, gamma=gamma, training=training)
        BaseModelMixin.__init__(self, name)

        assert isinstance(self.env.action_space, Discrete)
        assert isinstance(self.env.observation_space, Box)
        assert model_type in ('dense', 'conv', 'lstm')
        assert step_size == 1 or model_type == 'lstm'
        self.do_rollout = do_rollout
        self.gamma = gamma
        self.batch_size = batch_size
        self.training = training
        self.model_type = model_type
        self.model_params = model_params or {}
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.double_q = double_q
        self.dueling = dueling

        self.buffer = ReplayMemory(capacity=memory_capacity)

    @property
    def state_dim(self):
        # Returns: A list
        return [np.prod(list(self.env.observation_space.shape))]

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def init_target_q_net(self):
        self.sess.run([v_t.assign(v)
                       for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def _extract_network_params(self):
        net_params = {}

        net_class = dense_nn

        return net_class, net_params

    def create_q_networks(self):
        # The first dimension should have batch_size * step_size
        self.states = tf.placeholder(tf.float32, shape=(
            None, *self.state_dim), name='state')
        self.states_next = tf.placeholder(tf.float32, shape=(None, *self.state_dim),
                                          name='state_next')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='action')
        self.actions_next = tf.placeholder(
            tf.int32, shape=(None,), name='action_next')
        self.rewards = tf.placeholder(tf.float32, shape=(None,), name='reward')
        self.done_flags = tf.placeholder(
            tf.float32, shape=(None,), name='done')

        # The output is a probability distribution over all the actions.

        net_class, net_params = self._extract_network_params()

        if self.dueling:
            self.q_hidden = net_class(self.states, self.layer_sizes[:-1], name='Q_primary',
                                      **net_params)
            self.adv = dense_nn(self.q_hidden, self.layer_sizes[-1:] + [self.act_size],
                                name='Q_primary_adv')
            self.v = dense_nn(
                self.q_hidden, self.layer_sizes[-1:] + [1], name='Q_primary_v')

            # Average Dueling
            self.q = self.v + (self.adv - tf.reduce_mean(
                self.adv, reduction_indices=1, keep_dims=True))

            self.q_target_hidden = net_class(self.states_next, self.layer_sizes[:-1], name='Q_target',
                                             **net_params)
            self.adv_target = dense_nn(self.q_target_hidden, self.layer_sizes[-1:] + [self.act_size],
                                       name='Q_target_adv')
            self.v_target = dense_nn(self.q_target_hidden, self.layer_sizes[-1:] + [1],
                                     name='Q_target_v')

            # Average Dueling
            self.q_target = self.v_target + (self.adv_target - tf.reduce_mean(
                self.adv_target, reduction_indices=1, keep_dims=True))

        else:
            self.q = net_class(self.states, self.layer_sizes + [self.act_size], name='Q_primary',
                               **net_params)
            self.q_target = net_class(self.states_next, self.layer_sizes + [self.act_size],
                                      name='Q_target', **net_params)

        # The primary and target Q networks should match.
        self.q_vars = self.scope_vars('Q_primary')
        self.q_target_vars = self.scope_vars('Q_target')
        assert len(self.q_vars) == len(
            self.q_target_vars), "Two Q-networks are not same."

    def getQvalue(self, state):
        with self.sess.as_default():
            return self.q.eval({self.states: [state]})[-1]

    def build(self):
        self.create_q_networks()
        self.actions_selected_by_q = tf.argmax(
            self.q, axis=-1, name='action_selected')
        action_one_hot = tf.one_hot(
            self.actions, self.act_size, 1.0, 0.0, name='action_one_hot')
        pred = tf.reduce_sum(self.q * action_one_hot,
                             reduction_indices=-1, name='q_acted')

        if self.double_q:
            actions_next_flatten = self.actions_next + tf.range(
                0, self.batch_size * self.step_size) * self.q_target.shape[1]
            max_q_next_target = tf.gather(tf.reshape(
                self.q_target, [-1]), actions_next_flatten)
        else:
            max_q_next_target = tf.reduce_max(self.q_target, axis=-1)

        y = self.rewards + (1. - self.done_flags) * \
            self.gamma * max_q_next_target

        self.learning_rate = tf.placeholder(
            tf.float32, shape=None, name='learning_rate')
        self.loss = tf.reduce_mean(
            tf.square(pred - tf.stop_gradient(y)), name="loss_mse_train")
        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss, name="adam_optim")

        self.target_update = tf.group(
            [v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])

        with tf.variable_scope('summary'):
            q_summ = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.act_size):
                q_summ.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summ = tf.summary.merge(q_summ, 'q_summary')

            self.q_y_summ = tf.summary.histogram("batch/y", y)
            self.q_pred_summ = tf.summary.histogram("batch/pred", pred)
            self.loss_summ = tf.summary.scalar("loss", self.loss)

            self.ep_reward = tf.placeholder(tf.float32, name='episode_reward')
            self.ep_reward_summ = tf.summary.scalar(
                'episode_reward', self.ep_reward)

            self.merged_summary = tf.summary.merge_all(
                key=tf.GraphKeys.SUMMARIES)

        self.sess.run(tf.global_variables_initializer())
        self.init_target_q_net()

    def update_target_q_net(self):
        self.sess.run([v_t.assign(v)
                       for v_t, v in zip(self.q_target_vars, self.q_vars)])

    def act(self, state, epsilon=0.1):
        if self.training and np.random.random() < epsilon:
            return self.env.action_space.sample()

        with self.sess.as_default():
            return self.actions_selected_by_q.eval({self.states: [state]})[-1]

    ##########

    class TrainConfig(TrainConfig):
        lr = 0.001
        lr_decay = 1.0
        epsilon = 1.0
        epsilon_final = 0.02
        memory_capacity = 100000
        target_update_every_step = 100
        n_episodes = 500
        warmup_episodes = 450
        log_every_episode = 10
        update_per_sample = 1
        episode_length = 250

    def test(self):
        ob = self.env.reset(isTest=True)
        done = False
        step_counter = 0
        reward = 0.
        # print(f"Reset obs: {ob}")
        while not done and (step_counter < self.episode_length):
            a = self.act(self.obs_to_inputs(ob), 0)
            new_ob, r, done, info = self.env.step(a)
            step_counter += 1
            reward += r

            ob = new_ob
            # print(f"a: {a} in obs: {[f'{val:.3f}' for val in ob]}")
        return reward

    def update_networks(self, batch_data, feed_dict):
        if self.double_q:
            actions_next = self.sess.run(self.actions_selected_by_q, {
                self.states: batch_data['s_next']
            })
            feed_dict.update({self.actions_next: actions_next})

        _, q_val, q_target_val, loss, summ_str, _ = self.sess.run(
            [self.optimizer, self.q, self.q_target, self.loss,
                self.merged_summary, self.target_update],
            feed_dict
        )
        return summ_str

    def train(self, config: TrainConfig):
        self.episode_length = config.episode_length
        print("Config warmup_episodes: ", config.warmup_episodes)
        print("Config target_update_every_step: ",
              config.target_update_every_step)
        reward = 0.
        reward_history = [0.0]
        reward_averaged = []

        lr = config.lr
        eps = config.epsilon
        annealing_episodes = config.warmup_episodes or config.n_episodes
        eps_drop = (config.epsilon - config.epsilon_final) / annealing_episodes
        print("eps_drop:", eps_drop)
        step = 0

        """Warm start"""
        self.warmup_episodes = config.warmup_episodes
        if self.warmup_episodes:
            print("Warmstarting by loading a replay buffer")
            self.load_replay_buffer()
            start_time = time.time()
            average_time = []
            average_time_train = []
            average_time_sample = []
            # Training with a mini batch of samples!
            # //self.batch_size
            updates = self.warmup_episodes
            # updates = self.buffer.size//self.batch_size
            print("Going to train for %d steps (%.2f updates per sample)" %
                  (updates, config.update_per_sample))
            for _iter in range(updates):
                sample_time = time.time()
                batch_data = self.buffer.sample(self.batch_size)
                average_time_sample.append(time.time()-sample_time)
                feed_dict = {
                    self.learning_rate: lr,
                    self.states: batch_data['s'],
                    self.actions: batch_data['a'],
                    self.rewards: batch_data['r'],
                    self.states_next: batch_data['s_next'],
                    self.done_flags: batch_data['done'],
                    self.ep_reward: reward_history[-1],
                }

                train_time = time.time()
                self.update_networks(batch_data, feed_dict)
                # if _iter % config.target_update_every_step == 0:
                #     self.update_target_q_net()
                average_time_train.append(time.time()-train_time)
                if ((_iter % config.target_update_every_step) == 0) and _iter != 0:
                    time_for_10 = time.time()-start_time
                    average_time.append(copy.copy(time_for_10))
                    total_time = updates/config.target_update_every_step * \
                        np.mean(np.array(average_time))
                    time_left = (
                        updates-_iter)/config.target_update_every_step*np.mean(np.array(average_time))
                    print("%d/%d. Time to finish: %.1fmin (%ds). Total time required: %.1fmin (%ds). Average time for 10 updates: %.3fs. Time spent on training: %.1fms. Time for sample: %.1fms"
                          % (_iter, updates, time_left/60, time_left, total_time/60, total_time, np.mean(np.array(average_time)), np.mean(np.array(average_time_train))*1e3,
                             np.mean(np.array(average_time_sample))*1e3))
                    start_time = time.time()

            self.save_checkpoint(step=0)
            print("Testing warmstarted policy")
            rewards = []
            for _ in range(1):
                test_reward = self.test()
                rewards.append(test_reward)
            print(
                f"Warmstart test reward: {np.mean(rewards)} with rewards: {rewards}")
        # if not self.do_rollout:
        #     return 0
        # assert NotImplementedError
        best_reward = -99999
        for n_episode in range(config.n_episodes):
            ob = self.env.reset()
            done = False
            traj = []
            step_counter = 0
            while not done and (step_counter < config.episode_length):
                a = self.act(self.obs_to_inputs(ob), eps)
                new_ob, r, done, info = self.env.step(a)
                step_counter += 1
                step += 1
                reward += r

                traj.append(
                    Transition(self.obs_to_inputs(ob), a, r, self.obs_to_inputs(new_ob), done))
                ob = new_ob

                # No enough samples in the buffer yet.
                if self.buffer.size < self.batch_size:
                    continue

                # Training with a mini batch of samples!
                batch_data = self.buffer.sample(self.batch_size)
                feed_dict = {
                    self.learning_rate: lr,
                    self.states: batch_data['s'],
                    self.actions: batch_data['a'],
                    self.rewards: batch_data['r'],
                    self.states_next: batch_data['s_next'],
                    self.done_flags: batch_data['done'],
                    self.ep_reward: reward_history[-1],
                }

                summ_str = self.update_networks(batch_data, feed_dict)

                self.writer.add_summary(summ_str, step)
                if step % config.target_update_every_step == 0:
                    print("Updated q net")
                    self.update_target_q_net()
            print("%d: Done step_counter: %d. reward: %.1f, started pos at: [%.2f, %.2f]"
                  % (n_episode, step_counter, reward, self.env.base_pos_spawn_offset[0], self.env.base_pos_spawn_offset[1]))
            # Add all the transitions of one trajectory into the replay memory.
            self.buffer.add(traj)

            # save if best episode
            if n_episode % config.log_every_episode == 0 and n_episode != 0:
                # print("TESTING")
                test_reward = self.test()
                print(f"Test reward: {test_reward}")
                if test_reward > best_reward:
                    print("New best reward: %.2f" % test_reward)
                    self.save_checkpoint(step=n_episode)
                    best_reward = test_reward
            # print("N episode. %d" % n_episode)
            # One episode is complete.
            reward_history.append(int(reward))
            reward_averaged.append(np.mean(reward_history[-10:]))
            reward = 0.

            # Annealing the learning and exploration rate after every episode.
            lr *= config.lr_decay
            if eps > config.epsilon_final:
                eps = max(eps - eps_drop, config.epsilon_final)

            if reward_history and config.log_every_episode and n_episode % config.log_every_episode == 0:
                # Report the performance every `every_step` steps
                print(
                    "[episodes:{}/step:{}], best:{}, avg:{:.2f}:{}, lr:{:.4f}, eps:{:.4f}".format(
                        n_episode, step, np.max(reward_history),
                        np.mean(reward_history[-10:]), reward_history[-5:],
                        lr, eps, self.buffer.size
                    ))
                # self.save_checkpoint(step=step)
        self.save_checkpoint(step=step)

        print("[FINAL] episodes: {}, Max reward: {}, Average reward: {}".format(
            len(reward_history), np.max(reward_history), np.mean(reward_history)))

        data_dict = {
            'reward': reward_history,
            'reward_smooth10': reward_averaged,
        }
        plot_learning_curve(self.model_name, data_dict, xlabel='episode')

    def load_replay_buffer(self):
        's', 'a', 'r', 's_next', 'done'
        dir_path = os.path.dirname(os.path.realpath(__file__))

        dir_name = f"{dir_path}/../../../qlearning/offline_data"

        data = []
        len_replay_buffer = 0
        for file in os.listdir(dir_name):
            if file.endswith(".p"):
                file_name = os.path.join(dir_name, file)
                replay_buffer = pickle.load(open(file_name, "rb"))
                data.append(replay_buffer)
                len_replay_buffer += len(replay_buffer)
        print("Amount of data in replay buffer: %d" % len_replay_buffer)

        traj = []
        for replay_buffer in data:
            print("Amount of data loading into replay buffer: ", len(replay_buffer))
            for entry in replay_buffer:
                for _ in range(10):
                    self.buffer.add(Transition(
                        entry["obs"],
                        entry["action"],
                        entry["reward"],
                        entry["next_obs"],
                        entry["done"]
                    ))

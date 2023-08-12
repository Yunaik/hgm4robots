params = {
    'type': 'SAC',
    'universe': 'gym',
    'domain': "Valkyrie",
    'task': "v2",

    'log_dir': '~/ray_mbpo/',
    'exp_name': 'defaults',

    'kwargs': {
        "smooth_loss_weight": 2.0,
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,
        'n_initial_exploration_steps': 1000,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


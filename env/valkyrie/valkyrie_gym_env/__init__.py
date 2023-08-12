import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Valkyrie-v0',
    entry_point='valkyrie_gym_env.envs:Valkyrie',
)

register(
    id='Valkyrie-v4',
    entry_point='valkyrie_gym_env.envs:Valkyrie',
)

register(
    id='Valkyrie-v5',
    entry_point='valkyrie_gym_env.envs:Valkyrie_whole_body',
)

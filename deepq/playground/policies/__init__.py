from deepq.playground.policies.actor_critic import ActorCriticPolicy
from deepq.playground.policies.ddpg import DDPGPolicy
from deepq.playground.policies.dqn import DqnPolicy
from deepq.playground.policies.dqn_warmstart import DqnWarmstartPolicy
from deepq.playground.policies.ppo import PPOPolicy
from deepq.playground.policies.qlearning import QlearningPolicy
from deepq.playground.policies.reinforce import ReinforcePolicy

ALL_POLICIES = [
    ActorCriticPolicy,
    DDPGPolicy,
    DqnPolicy,
    PPOPolicy,
    QlearningPolicy,
    ReinforcePolicy,
    DqnWarmstartPolicy
]

# from playground.policies.actor_critic import ActorCriticPolicy
# from playground.policies.ddpg import DDPGPolicy
# from playground.policies.dqn import DqnPolicy
# from playground.policies.ppo import PPOPolicy
# from playground.policies.qlearning import QlearningPolicy
# from playground.policies.reinforce import ReinforcePolicy

# ALL_POLICIES = [
#     ActorCriticPolicy,
#     DDPGPolicy,
#     DqnPolicy,
#     PPOPolicy,
#     QlearningPolicy,
#     ReinforcePolicy
# ]

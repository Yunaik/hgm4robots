import os
from qlearning.getPolicy import getOps
import gym
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import warnings
warnings.filterwarnings("ignore")

CUR_DIR = os.path.join(os.path.dirname(__file__))

image_list = []
image_list2 = []

getAction, getQ = getOps("valkyrie-v5",
                         checkpoint_dir=f"{CUR_DIR}/working_policy/checkpoints/valkyrie-v5",
                         config_dir=f"{CUR_DIR}/working_policy/val-v5")

render = True
save_video = True
save_tf = True

env = gym.make("Valkyrie-v5",
               incremental_control=False,
               incremental_control_high=False,
               renders=render,
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
               base_pos_spawn_offset=[0.20, 0.0, 0],
               print_reward_details=False,
               filter_action=True,
               action_bandwidth=9,
               load_obstacle=0,  # 0 is none, 1 is box, 2 is incline, 3 is slippery
               exertForce=False,  # set to True for push disturbance
               fixed_disturbance=True,
               maxImpulse=100,
               save_trajectories=True,
               high_level_frequency=0.5,
               )

env.frames = []

env.links_to_read = None
obs = env.reset()
reward = 0
for i in range(int(0.5*80)):
    # epsilon = 0.0 equals deterministic policy
    a = getAction(obs, epsilon=0.0)
    obs, r, done, reward_info = env.step(a)
    reward += r

    if save_video:
        image_list.append(env.render(distance=7, yaw=45,
                                     pitch=0, roll=0,))  # standard
        image_list2.append(env.render(distance=5, yaw=0,
                                      pitch=0, roll=0,))  # standard
    if done:
        break

print("Reward final: ", reward)
if save_video:
    clip = ImageSequenceClip(image_list, fps=25)
    clip.write_videofile('video.mp4', fps=25, audio=False)

    clip = ImageSequenceClip(image_list2, fps=25)
    clip.write_videofile('video2.mp4', fps=25, audio=False)

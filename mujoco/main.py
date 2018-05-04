import tensorflow as tf
import numpy as np
import gym
from models import ActorCritic as model
import time
import os


def main():
    random_seed = 1
    tf.set_random_seed(random_seed)
    env_name = 'Hopper-v2'
    np.random.seed(random_seed)
    env = gym.make(env_name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    max_path_length = None
    max_path_length = max_path_length or env.spec.max_episode_steps
    print('max_path_length:', max_path_length)

    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    print('ob_dim:', ob_dim)
    print('ac_dim', ac_dim)
    gamma = 0.99
    gae_lambda = 0.98
    actor_learning_rate = 0.005
    critic_learning_rate = 0.001

    animate = True
    restore = True
    n_iter = 10
    min_timesteps_per_batch = 100

    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = 'Hopper-v2' + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print(logdir)

    tf.reset_default_graph()
    my_model = model(ob_dim, ac_dim, discrete, gamma, gae_lambda, actor_learning_rate, critic_learning_rate
                     )
    my_model.train(env_name, env, restore, animate, n_iter, max_path_length, min_timesteps_per_batch,
                   os.path.join(logdir, '%d' % random_seed))


if __name__ == '__main__':
    main()

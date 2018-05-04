import tensorflow as tf
import numpy as np
import time
import logz
import os
import inspect


class ActorCritic(object):
    def __init__(self, ob_dim, ac_dim, discrete, gamma, gae_lambda, actor_learning_rate, critic_learning_rate
                 ):
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.observation_placeholder = tf.placeholder(shape=[None, ob_dim], name='observation', dtype=tf.float32)
        if self.discrete:
            self.action_target_placeholder = tf.placeholder(shape=[None], name='action', dtype=tf.int32)
        else:
            self.action_target_placeholder = tf.placeholder(shape=[None, ac_dim], name='action', dtype=tf.float32)
        self.advantages_placeholder = tf.placeholder(shape=[None], name='advantages', dtype=tf.float32)

        self.actor_nn_init(actor_learning_rate)
        self.critic_nn_init( critic_learning_rate)

    def build_nn(self, input_placeholder, output_size, scope_name):
        with tf.name_scope(scope_name):
            dense = input_placeholder
            dense1 = tf.layers.dense(inputs=dense, units=self.ob_dim * 10, activation=tf.nn.tanh)
            dense2 = tf.layers.dense(inputs=dense1, units=int(np.mean([self.ob_dim,self.ac_dim])) * 10, activation=tf.nn.tanh)
            dense3 = tf.layers.dense(inputs=dense2, units=self.ac_dim * 10, activation=tf.nn.tanh)
            # dense1 = tf.layers.dense(inputs=dense, units=64, activation=tf.nn.tanh)
            # dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.tanh)
            # dense3 = tf.layers.dense(inputs=dense2, units=64, activation=tf.nn.tanh)
            output = tf.layers.dense(inputs=dense3, units=output_size, activation=None)
        return output

    def actor_nn_init(self, actor_learning_rate):
        self.actor_logits = self.build_nn(self.observation_placeholder,  output_size=self.ac_dim,
                                          scope_name='actor_nn')

        if self.discrete:
            self.sampled_actions_prob = tf.reshape(tf.multinomial(self.actor_logits, 1), [-1])
            actor_logprob = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.action_target_placeholder,
                                                                             logits=self.actor_logits)
        else:
            logstd = tf.Variable(tf.zeros([1, self.ac_dim]), name="logstd", dtype=tf.float32)
            std = tf.exp(logstd)
            uniform_distribution = tf.random_normal(tf.shape(self.actor_logits))
            self.sampled_actions_prob = self.actor_logits + std * uniform_distribution
            actor_logprob = -0.5 * tf.reduce_sum(tf.square((self.actor_logits - self.action_target_placeholder) / std),
                                                 axis=1)
        loss = - tf.reduce_mean(actor_logprob * self.advantages_placeholder)
        self.actor_update_optimizer = tf.train.AdamOptimizer(learning_rate=actor_learning_rate).minimize(loss)

    def critic_nn_init(self, critic_learning_rate):
        self.critic_prediction = tf.squeeze(
            self.build_nn(self.observation_placeholder, output_size=1, scope_name='critic_nn'))
        self.critic_target_placeholder = tf.placeholder(shape=[None], name='critic_target', dtype=tf.float32)
        loss = tf.nn.l2_loss(self.critic_prediction - self.critic_target_placeholder)
        self.critic_update_optimizer = tf.train.AdamOptimizer(critic_learning_rate).minimize(loss)

    def train(self,env_name, env, restore, animate, n_iter, max_path_length, min_timesteps_per_batch, logdir):
        # start = time.time()

        logz.configure_output_dir(logdir)

        # Log experimental parameters
        args = inspect.getargspec(self.train)[0]
        locals_ = globals()
        params = {k: locals_[k] if k in locals_ else None for k in args}
        logz.save_params(params)
        saver = tf.train.Saver()

        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)

        self.sess.__enter__()  # equivalent to `with sess:`

        model_path = './'+env_name+'/model.ckpt'
        # if not os.path.exists(model_path):
        #     saver.save(self.sess, model_path)
        #     print("Model saved in path: %s" % model_path)

        if not restore:
            tf.global_variables_initializer().run()  # pylint: disable=E1101
        else:
            saver.restore(self.sess, model_path)
            print("Model restored from path: %s" % model_path)
        for iter in range(n_iter):
            critic, advantages, paths = self.pre_play(env, animate, iter, max_path_length, min_timesteps_per_batch)
            # print('observation', self.observation_update)
            self.sess.run(self.critic_update_optimizer, feed_dict={self.observation_placeholder: self.observation_update,
                                                                   self.critic_target_placeholder: self.normalize(critic)})
            self.sess.run(self.actor_update_optimizer, feed_dict={self.observation_placeholder: self.observation_update,
                                                                  self.action_target_placeholder: self.action_update,
                                                                  self.advantages_placeholder: self.normalize(advantages)})
            returns = [path['reward'].sum() for path in paths]
            ep_lengths = [len(path['reward']) for path in paths]

            # logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", iter)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.dump_tabular()
            logz.pickle_tf_vars()
            if (iter + 1) % 50 == 0:
                saver.save(self.sess, model_path)
                print("Model saved in path: %s" % model_path)
        saver.save(self.sess, model_path)
        print("Model saved in path: %s" % model_path)


    def calculate_critic(self, paths):
        q_paths = []
        for path in paths:
            q = 0
            q_path = []
            for reward in reversed(path['reward']):
                q = reward + self.gamma * q
                q_path.append(q)
            q_path.reverse()
            q_paths.extend(q_path)

        critic = self.sess.run(self.critic_prediction,
                               feed_dict={self.observation_placeholder: self.observation_update})
        critic = self.normalize(critic, np.mean(q_paths), np.std(q_paths))

        adv_paths = []
        index = 0
        for path in paths:
            adv = 0
            adv_path = []
            v_next = 0
            index += len(path['reward'])

            for reward, v in zip(reversed(path['reward']), critic[index-1:None:-1]):
                q = reward + self.gamma * v_next - v
                adv = q + self.gae_lambda * self.gamma * adv
                adv_path.append(adv)
                v_next = v
            adv_path.reverse()
            adv_paths.extend(adv_path)
        critic = critic + adv_paths
        return critic, adv_paths

    def pre_play(self, env, animate, iter, max_path_length, min_timesteps_per_batch):
        total_timesteps = 0
        timesteps = 0
        paths = []
        while True:
            # simulate 1 episode and get a path within limited steps
            ob = env.reset()
            obs, actions, rewards = [], [], []
            # animate = (len(paths) == 0 and (iter % 10 == 0) and animate)
            steps = 0
            while True:
                if animate:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                act = self.sess.run(self.sampled_actions_prob, feed_dict={self.observation_placeholder: [ob]})
                act = act[0]
                actions.append(act)

                ob, reward, done, _ = env.step(act)
                rewards.append(reward)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {'observation': np.array(obs),
                    'action': np.array(actions),
                    'reward': np.array(rewards)}
            paths.append(path)
            timesteps += 1
            if timesteps > min_timesteps_per_batch:
                break
        total_timesteps += timesteps

        self.observation_update = np.concatenate([path['observation'] for path in paths])
        self.action_update = np.concatenate([path['action'] for path in paths])
        critic, advantages = self.calculate_critic(paths)
        return critic, advantages, paths

    def normalize(self, data, mean=0.0, std=1.0):
        n_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        return n_data * (std + 1e-8) + mean

#!/usr/bin/env python
#some of this code was borrowed from https://github.com/jj-zhu/jadagger/blob/master/run_dagger.py #

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import nel
import greedy_policy
from canonical_plot import plot

def trans_state(observation):
    moved_dict={'True':1,'False':0}
    s1=np.array([moved_dict[str(observation['moved'])]])
    s2=observation['scent']
    s3=observation['vision']
    s3=s3.ravel()
    state=np.append(s1,s2)
    state=np.append(state,s3)
    return state


def policy_fn(env, vision):
    n_tongs = env._agent._items[1]  # number of tongs carried
    q_value = greedy_policy.value_iteration(vision, n_tongs)
    action = np.argmax(q_value[5, 5, 0] + 1e-2 * np.random.rand(3))  # add noise to break ties
    return int(action)
def policy_fn_1(n_tongs, vision):
    q_value = greedy_policy.value_iteration(vision, n_tongs)
    action = np.argmax(q_value[5, 5, 0] + 1e-2 * np.random.rand(3))  # add noise to break ties
    return int(action)
    
#===========================================================================
# generate expert data
#===========================================================================
# param
#    expert_policy_file = 'experts/Humanoid-v1.pkl'
envname = 'NEL-v0'
render = 0
num_rollouts = 1
max_timesteps = 10000
n_steps = 10000
# policy_fn contains expert policy
#    policy_fn = load_policy.load_policy(expert_policy_file)
   
with tf.Session():
    tf_util.initialize()
    import gym
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(env, obs['vision'])
            # action using expert policy policy_fn
            observations.append(trans_state(obs))
            actions.append([action])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    
    # pass observations, actions to imitation learning
    obs_data = np.squeeze(np.array(observations))
    act_data = np.array(actions)#np.squeeze(np.array(actions))
    
save_expert_mean = np.mean(returns)
save_expert_std = np.std(returns)

#===========================================================================
# set up the network structure for the imitation learning policy function
#===========================================================================
# dim for input/output
#    print obs_data
#print act_data
#    print act_data.shape
obs_dim = obs_data.shape[1]
act_dim = act_data.shape[1]

# architecture of the MLP policy function
x = tf.placeholder(tf.float32, shape=[None, obs_dim])
yhot = tf.placeholder(tf.float32, shape=[None, act_dim])

h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu)
h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu)
yhat = tf.layers.dense(inputs=h3, units=act_dim, activation=None)

loss_l2 = tf.reduce_mean(tf.square(yhot - yhat))
train_step = tf.train.AdamOptimizer().minimize(loss_l2)

#===========================================================================
# run DAgger alg
#===========================================================================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # record return and std for plotting
    save_mean = []
    save_std = []
    save_train_size = []
    # loop for dagger alg
    for i_dagger in xrange(20):
        print 'DAgger iteration ', i_dagger
        # train a policy by fitting the MLP
        batch_size = 25
        for step in range(10000):
            batch_i = np.random.randint(0, obs_data.shape[0], size=batch_size)
            train_step.run(feed_dict={x: obs_data[batch_i, ], yhot: act_data[batch_i, ]})
            if (step % 1000 == 0):
                print 'opmization step ', step
                print 'obj value is ', loss_l2.eval(feed_dict={x:obs_data, yhot:act_data})
        print 'Optimization Finished!'
        # use trained MLP to perform
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        observations_trans = []
        actions = []
        ntongs = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            obs_trans = trans_state(obs)
            done = False
            totalr = 0.
            steps = 0
#            while not done:
            for _ in range(n_steps):
                action = np.rint(yhat.eval(feed_dict={x:obs_trans[None, :]})[0][0]).astype(int)
#                print action
                observations.append(obs)
                ntongs.append(env._agent._items[1])
                observations_trans.append(obs_trans)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                obs_trans = trans_state(obs)
                totalr += r
                steps += 1   
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # expert labeling
        act_new = []
        for i_label in xrange(len(observations)):
            act_new.append([policy_fn_1(ntongs[i_label], observations[i_label]['vision'])])#[None, :]))
        # record training size
        train_size = obs_data.shape[0]
        # data aggregation
        obs_data = np.concatenate((obs_data, np.array(observations_trans)), axis=0)
        act_data = np.concatenate((act_data, np.array(act_new)), axis=0)#np.squeeze(np.array(act_new))), axis=0)
#        print act_data
        # record mean return & std
        save_mean = np.append(save_mean, np.mean(returns))
        save_std = np.append(save_std, np.std(returns))
        save_train_size = np.append(save_train_size, train_size)
        ############################################################
    n_steps=10000
    # Returns the average reward over n_test_episodes of trained model
    ## and standard deviation
    rewards=[]
    state = env.reset()
    state =  trans_state(state)
    #    state = np.expand_dims(state,0)
    for step in range(n_steps):
#        env.render()
        action = np.rint(yhat.eval(feed_dict={x:state[None, :]})[0][0]).astype(int)
#        print action
        next_state, reward, done, info = env.step(action)
        next_state =  trans_state(next_state)
    #        next_state = np.expand_dims(next_state,0)
        state = next_state
        
        rewards.append(reward)        
dagger_results = {'means': save_mean, 'stds': save_std, 'train_size': save_train_size,
                  'expert_mean':save_expert_mean, 'expert_std':save_expert_std}
print 'DAgger iterations finished!'



 

print dagger_results
#plot("dagger_tr",tr_rewards)
plot("dagger_pr",rewards)
filename = "dagger_rewards_file"
with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ rewards, dagger_results], f,protocol=2)  


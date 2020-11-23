# -*- coding: utf-8 -*
import os
import numpy as np
import pickle
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym import wrappers
from collections import deque
import itertools

from Config import Config, DDQNConfig, DQfDConfig
from DQfD_sas import DQfD
from expert import value_iteration as expert_VI
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

# ===================================================== #
#                    train with DQfD                    #
# ===================================================== #
def run_DQfD(index, env, file_demo, file_name):
    with open(file_demo+'demo.p', 'rb') as f:
        demo_transitions = pickle.load(f)
        demo_transitions = deque(itertools.islice(demo_transitions, 0, DQfDConfig.demo_buffer_size))
        assert len(demo_transitions) == DQfDConfig.demo_buffer_size
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD(env, DQfDConfig(), demo_transitions=demo_transitions)

    agent.pre_train()  # use the demo_sas data to pre-train network

    REWARDS, REWARD100, episode, replay_full_episode = [], [], 0, None
    reward100, n_step_reward, state = 0, None, env.reset()
    state = trans_state(state)
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)
    next_state =  trans_state(next_state)
    reward100 += reward
    REWARDS.append(reward)
    sas_state = np.hstack((state, np.array([action]), next_state))
    sas_state = sas_state.ravel()

    t_q = deque(maxlen=DQfDConfig.trajectory_n)
    for steps in range(DQfDConfig.episode):
        action = agent.egreedy_action(sas_state)  # e-greedy action for train
        next_next_state, reward, done, _ = env.step(action)
        next_next_state = trans_state(next_next_state)
        reward100 += reward
        REWARDS.append(reward)

        state = next_state
        next_state = next_next_state
        next_sas_state = np.hstack((state, np.array([action]), next_state))
        next_sas_state = next_sas_state.ravel()

        t_q.append([sas_state, action, reward, next_sas_state, done, 0.0])
        sas_state = next_sas_state

        # record the earliest reward for the sub-sequence
        if len(t_q) < t_q.maxlen: 
            reward_to_sub = 0.
        else: 
            reward_to_sub = t_q[0][2]
            if n_step_reward is None:  # only compute once when t_q first filled
                n_step_reward = sum([t[2]*DQfDConfig.GAMMA**i for i, t in enumerate(t_q)])
            else:
                n_step_reward = (n_step_reward - reward_to_sub) / DQfDConfig.GAMMA
                n_step_reward += reward*DQfDConfig.GAMMA**(DQfDConfig.trajectory_n-1)

            t_q[0].extend([n_step_reward, next_sas_state, done, t_q.maxlen])  # actual_n is max_len here
            update_eps=True if (steps+1)%DQfDConfig.eps_gap==0 else False
            agent.perceive(t_q[0], update_eps=update_eps)  # perceive when a transition is completed
            if (steps+1)%DQfDConfig.UPDATE_ESTIMATE_NET==0:
                agent.train_Q_network(update=False)  # train along with generation
            replay_full_episode = replay_full_episode or episode

        if (steps+1) % DQfDConfig.UPDATE_TARGET_NET == 0:
            if agent.replay_memory.full(): agent.sess.run(agent.update_target_net)
            
        if (steps+1) % DQfDConfig.eps_gap == 0:
            episode += 1
            if replay_full_episode is not None:
                print("episode: {}  trained-episode: {}  reward100: {}  memory length: {}  epsilon: {}"
                      .format(episode, episode-replay_full_episode, reward100, len(agent.replay_memory), agent.epsilon))
            REWARD100.append(reward100)
            reward100 = 0

        if (steps+1) % (DQfDConfig.eps_gap*100) == 0:
            with open(file_name+'REWARD100.p', 'wb') as f: pickle.dump(REWARD100, f, protocol=2)
            with open(file_name+'REWARD100.txt', 'wb') as f: f.write(str(REWARD100))
            with open(file_name+'REWARDS.p', 'wb') as f: pickle.dump(REWARDS, f, protocol=2)
            with open(file_name+'REWARDS.txt', 'wb') as f: f.write(str(REWARDS))
            plot(1, REWARDS, file_name)

    with open(file_name+'REWARD100.p', 'wb') as f: pickle.dump(REWARD100, f, protocol=2)
    with open(file_name+'REWARD100.txt', 'wb') as f: f.write(str(REWARD100))
    with open(file_name+'REWARDS.p', 'wb') as f: pickle.dump(REWARDS, f, protocol=2)
    with open(file_name+'REWARDS.txt', 'wb') as f: f.write(str(REWARDS))
    plot(1, REWARDS, file_name)

# ===================================================== #
#              generate demonstration data              #
# ===================================================== #
# extend [n_step_reward, n_step_away_state]
def set_n_step(container, n):
    t_list = list(container)
    # accumulated reward of first (trajectory_n-1) transitions
    n_step_reward = sum([t[2] * DQfDConfig.GAMMA**i for i, t in enumerate(t_list[0:min(len(t_list), n) - 1])])
    for begin in range(len(t_list)):
        end = min(len(t_list) - 1, begin + DQfDConfig.trajectory_n - 1)
        n_step_reward += t_list[end][2]*DQfDConfig.GAMMA**(end-begin)
        # extend[n_reward, n_next_s, n_done, actual_n]
        t_list[begin].extend([n_step_reward, t_list[end][3], t_list[end][4], end-begin+1])
        n_step_reward = (n_step_reward - t_list[begin][2])/DQfDConfig.GAMMA
    return t_list

def get_demo_greedy(env, file_name):
    print(DQfDConfig.demo_buffer_size)
    demo_buffer = deque()
    demo_sas_buffer = deque()
    demo = []
    demo_sas = []
    REWARDS, REWARD100, reward100 = [], [], 0
    state = env.reset()
    vision = state['vision']
    n_tongs = env._agent._items[1]  # number of tongs carried
    q_value = expert_VI(vision, n_tongs)
    action = np.argmax(q_value[5, 5, 0] + 1e-2 * np.random.rand(3))  # add noise to break ties
    state = trans_state(state)
    next_state, reward, done, _ = env.step(action)
    reward100 += reward
    REWARDS.append(reward)
    vision = next_state['vision']
    n_tongs = env._agent._items[1]
    q_value = expert_VI(vision, n_tongs)
    action = np.argmax(q_value[5, 5, 0] + 1e-2 * np.random.rand(3))
    next_state = trans_state(next_state)
    # demo.append([state, action, reward, next_state, done, 1.0])

    sas_state = np.hstack((state, np.array([action]), next_state))
    sas_state = sas_state.ravel()
    state = next_state

    for steps in range(DQfDConfig.demo_buffer_size):
        next_state, reward, done, _ = env.step(action)
        reward100 += reward
        REWARDS.append(reward)
        vision = next_state['vision']
        n_tongs = env._agent._items[1]
        q_value = expert_VI(vision, n_tongs)
        action = np.argmax(q_value[5, 5, 0] + 1e-2 * np.random.rand(3))
        next_state = trans_state(next_state)
        demo.append([state, action, reward, next_state, done, 1.0])

        next_sas_state = np.hstack((state, np.array([action]), next_state))
        next_sas_state = next_sas_state.ravel()
        demo_sas.append([sas_state, action, reward, next_sas_state, done, 1.0])
        state = next_state
        sas_state = next_sas_state

        if (steps+1)%DQfDConfig.eps_gap == 0:
            print("demo - steps: {}  reward100: {}".format(steps, reward100))
            REWARD100.append(reward100)
            reward100 = 0

        if (steps+1)%(DQfDConfig.eps_gap*10) == 0:
            with open(file_name+'REWARD100.p', 'wb') as f: pickle.dump(REWARD100, f, protocol=2)
            with open(file_name+'REWARD100.txt', 'wb') as f: f.write(str(REWARD100))
            with open(file_name+'REWARDS.p', 'wb') as f: pickle.dump(REWARDS, f, protocol=2)
            with open(file_name+'REWARDS.txt', 'wb') as f: f.write(str(REWARDS))
            plot(1, REWARDS, file_name)

    with open(file_name+'REWARD100.p', 'wb') as f: pickle.dump(REWARD100, f, protocol=2)
    with open(file_name+'REWARD100.txt', 'wb') as f: f.write(str(REWARD100))
    with open(file_name+'REWARDS.p', 'wb') as f: pickle.dump(REWARDS, f, protocol=2)
    with open(file_name+'REWARDS.txt', 'wb') as f: f.write(str(REWARDS))
    plot(1, REWARDS, file_name)

    demo = set_n_step(demo, DQfDConfig.trajectory_n)
    demo_buffer.extend(demo)
    with open(file_name+'demo_s.txt', "w") as file: file.write(str(demo_buffer))
    with open(file_name+'demo_s.p', 'wb') as f: pickle.dump(demo_buffer, f, protocol=2)

    demo_sas = set_n_step(demo_sas, DQfDConfig.trajectory_n)
    demo_sas_buffer.extend(demo_sas)
    with open(file_name+'demo.txt', "w") as file: file.write(str(demo_sas_buffer))
    with open(file_name+'demo.p', 'wb') as f: pickle.dump(demo_sas_buffer, f, protocol=2)


# ===================================================== #
#                          main                         #
# ===================================================== #
if __name__ == '__main__':
    folder = './results_greedy/'
    if not os.path.exists(folder): os.makedirs(folder)
    env = gym.make(DQfDConfig.ENV_NAME)
    
    file_demo = folder + 'dqfd_sas_'
    # get_demo_greedy(env, file_demo)
    
    file_name = folder + 'dqfd_sas_trial' + str(DQfDConfig.trial_num) + '_'
    run_DQfD(DQfDConfig.trial_num, env, file_demo, file_name)

    env.close()
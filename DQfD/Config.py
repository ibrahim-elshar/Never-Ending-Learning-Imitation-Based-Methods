# -*- coding: utf-8 -*- 
import os


class Config:
    ENV_NAME = 'NEL-render-v0'
    # ENV_NAME = "CartPole-v0"
    GAMMA = 1.0  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    LEARNING_RATE = 0.001
    eps_gap = 100

    DEMO_RATIO = 0.1
    BATCH_SIZE = 64

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # trial_num = 1
    # - BATCH_SIZE:         size of minibatch
    # - LAMBDA:             the weights for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    # - UPDATE_TARGET_NET:  update eval_network params every {} steps
    # -------------------------- trial 0 ------------------------ #
    # BATCH_SIZE = 64
    # LAMBDA = [1.0, 0.0, 1.0, 10e-5]
    # UPDATE_TARGET_NET = 1000
    # -------------------------- trial 1 ------------------------ #
    # BATCH_SIZE = 32
    # LAMBDA = [1.0, 0.3, 1.0, 10e-5]
    # UPDATE_TARGET_NET = 500
    # -------------------------- trial 2 ------------------------ #
    # BATCH_SIZE = 32
    # LAMBDA = [1.0, 0.5, 0.5, 10e-5]
    # UPDATE_TARGET_NET = 500
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    START_TRAINING = 1000 # 1000 # experience replay buffer size
    PRETRAIN_STEPS = 10000  # 750000
    demo_buffer_size = 10000
    replay_buffer_size = demo_buffer_size * 5
    
    episode = 2000000

    trajectory_n = 10  # for n-step TD-loss (both demo data and generated data)

    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo.p')


class DDQNConfig(Config):
    demo_mode = 'get_demo'
    # PRETRAIN_STEPS = 100  # 750000
    # demo_buffer_size = 100
    # replay_buffer_size = demo_buffer_size * 10


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)
    trial_num = 1
    LAMBDA = [1.0, 0.3, 1.0, 10e-5]
    demo_buffer_size = 10000
    replay_buffer_size = demo_buffer_size * 5
    UPDATE_ESTIMATE_NET = 100
    UPDATE_TARGET_NET = 1000

class DQfDConfig2(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)
    trial_num = 2
    LAMBDA = [1.0, 1.0, 0.5, 10e-5]
    demo_buffer_size = 10000
    replay_buffer_size = demo_buffer_size * 5
    UPDATE_ESTIMATE_NET = 100
    UPDATE_TARGET_NET = 1000

class DQfDConfig3(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)
    trial_num = 3
    LAMBDA = [1.0, 0.3, 1.0, 10e-5]
    demo_buffer_size = 50000
    replay_buffer_size = demo_buffer_size * 2
    UPDATE_ESTIMATE_NET = 10
    UPDATE_TARGET_NET = 1000
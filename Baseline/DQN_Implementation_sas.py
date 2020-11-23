#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:54:24 2018
@author: ibrahim, mharding
"""
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import collections, random
import matplotlib.pyplot as plt
import math 
import nel
from canonical_plot import plot
import pickle

def trans_state(observation):
    moved_dict={'True':1,'False':0}
    s1=np.array([moved_dict[str(observation['moved'])]])
    s2=observation['scent']
    s3=observation['vision']
    s3=s3.ravel()
    state=np.append(s1,s2)
    state=np.append(state,s3)
    return state


#hyperparameters
hidden_layer1 = 735 
hidden_layer2 = 735 
hidden_layer3 = 735 
dueling_hidden_layer3 = 128  
lr = 0.001
gamma =  0.99
burn_in = 10000 
mem_size = 50000
initial_epsilon = 1 
final_epsilon = 0.1 
exploration_decay_steps = 1*10**5 
num_episodes = 2000000
minibatch_size = 32
save_weights_num_episodes = 1000
steps_update_target_network_weights = 2000
k_steps_before_minibatch = 1

    

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    # Define mapping from environment name to 
    # list containing [state shape, n_actions, lr]

    def __init__(self, env, is_double=False):
        # DQN network is instantiated using Keras
        self.STATE_DIMS = (735,)#(367,)#env.observation_space.shape[0]
        self.Q_DIMS = env.action_space.n
#        STATE_DIMS = (367,)
#        Q_DIMS = 3
        Layers=[
        keras.layers.Dense(hidden_layer1, input_shape=self.STATE_DIMS, activation ='relu'),
        keras.layers.Dense(hidden_layer2, activation='relu'),
        keras.layers.Dense(hidden_layer3, activation='relu'),
        keras.layers.Dense(self.Q_DIMS, activation='linear') 
        ]
        self.model = keras.models.Sequential(Layers)
        self.model.compile(loss='mse', 
                optimizer=keras.optimizers.Adam(lr=lr))

    def predict(self, state, **kwargs):
        # Return network predicted q-values for given state
        return self.model.predict(state, **kwargs)

    def fit(self, pred_values, true_values, **kwargs):
        # Fit the model we're training according to fit() API
        return self.model.fit(pred_values, true_values, **kwargs )

    def set_weights(self, *args, **kwargs):
        # Set weights of model according to set_weights Keras API
        return self.model.set_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        # Get weights of model according to set_weights Keras API
        return self.model.get_weights(*args, **kwargs)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(suffix)

    def load_model(self, model_file):
		# Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(weight_file)

class Dueling_QNetwork(QNetwork):
    # Define mapping from environment name to 
    # list containing [state shape, n_actions, lr]

    def __init__(self, env):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        self.STATE_DIMS = env.observation_space.shape[0]
        self.Q_DIMS = env.action_space.n        
        
        inputs = keras.layers.Input(shape=self.STATE_DIMS)
        penult_dqn_layer = None

        h0_out = keras.layers.Dense(hidden_layer1, activation='relu')(inputs)
        h1_out = keras.layers.Dense(hidden_layer2, activation='relu')(h0_out)
        penult_dqn_layer = h1_out
        # We need to diverge the network architecture into 2 fully connected
        ## streams from the output of the h1
        # First, the state-value stream: a fully-connected layer of 128 units
        ## which is then passed through to a scalar output layer
        penult_dqn_out_vs = keras.layers.Dense(dueling_hidden_layer3, activation='relu')(penult_dqn_layer)
        value_out = keras.layers.Dense(1, activation='relu')(penult_dqn_out_vs)
        # In parallel, next, the advantage-value stream: similarly a fc layer of 128 units
        ## then passed to another fc layer with output size = Q_dim
        h2_out_advs = keras.layers.Dense(dueling_hidden_layer3, activation='relu')(penult_dqn_layer)
        adv_out  = keras.layers.Dense(self.Q_DIMS, activation='relu')(h2_out_advs)

        # Lastly, the output of the Dueling network is defined as a function
        ## of the two streams:
        ## Q_vals = value_out - f(adv_out)  // Using broadcasting from TF
        ## where f(adv_out) = adv_out - sample_avg(adv)
        sample_avg_adv = keras.layers.Lambda(\
                                lambda l_in: keras.backend.mean(l_in))(adv_out)
        f_adv = keras.layers.Lambda(lambda l_in: l_in[0]-l_in[1])\
                                    ([adv_out, sample_avg_adv])
        Q_vals = keras.layers.Lambda(lambda l_in: l_in[0]-l_in[1])\
                                    ([value_out, f_adv])
        self.model = keras.models.Model(inputs=inputs, outputs=Q_vals)
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=lr))

class Replay_Memory():

    def __init__(self, burn_in, memory_size=mem_size):
        # The memory essentially stores transitions recorded from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        batch = random.sample(self.memory, batch_size)
        return batch

    def append(self, transition):
        # Appends transition to the memory.     
        self.memory.append(transition)

class Deep_Agent():
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #        (a) Epsilon Greedy Policy.
    #         (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, model_name, render=False, num_episodes=10000, curve_episodes=200):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env = gym.make(environment_name)

        # Instantiate the models
        self.is_DDQN = False
        if model_name == "dqn" or model_name == 'ddqn':
            if model_name == "dqn": 
                print("DQN model")
            if model_name == "ddqn":
                print("DDQN model")
                self.is_DDQN = True
            self.model_name = model_name
            self.model = QNetwork(self.env, is_double = self.is_DDQN)
            self.model_target = QNetwork(self.env, is_double = self.is_DDQN)
            self.model_target.set_weights(self.model.get_weights())
        else:
            self.model_name = "dueling"
            self.model = Dueling_QNetwork(self.env)
            self.model_target = Dueling_QNetwork(self.env) 
            self.model_target.set_weights(self.model.get_weights())
            print("Dueling model")

        self.step_rewards =[]
        self.burn_in = burn_in
        self.memory = Replay_Memory(self.burn_in)
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.epsilon = self.initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_decay_steps = exploration_decay_steps
        self.num_episodes = num_episodes
        self.minibatch_size = minibatch_size
        self.n_steps_before_update_tgt = steps_update_target_network_weights 
        self.k_steps_before_minibatch = k_steps_before_minibatch
        self.num_of_episodes_to_update_train_and_perf_curve = curve_episodes

        self.avg_training_episodes_return=[]
        self.avg_performance_episodes_return = []

    def epsilon_greedy_policy(self, q_values, force_epsilon=None):
        # Creating epsilon greedy probabilities to sample from.             
        eps = None
        if force_epsilon:
            eps = force_epsilon
        else:
            # Decay epsilon, save and use
            eps = max(self.final_epsilon, self.epsilon - (self.initial_epsilon - self.final_epsilon)/self.exploration_decay_steps)
            self.epsilon = eps
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        action = np.argmax(q_values)
        return action


    def fast_minibatch_update(self, replay_batch, update_tgt=False):
        # Split up axes of replay_batch buffer
        states = np.array(map(lambda bs: bs[0], replay_batch))
        actions = np.array(map(lambda bs: bs[1], replay_batch))
        rewards = np.array(map(lambda bs: bs[2], replay_batch))
        next_states = np.array(map(lambda bs: bs[3], replay_batch))
        done_flags = np.array(map(lambda bs: bs[4], replay_batch))

        # For states, next_states arrays, they have an additional dimension 
        ## due to how they are saved, so we must squeeze them down one dim
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Gather target q_values from replay batch
        batch_q_values_arr= self.model.predict(states)
        tgt_q_values = None
        if self.is_DDQN:
            selected_actions = np.argmax(self.model.predict(next_states), axis=1)
            # Outputs a vector tgt_q_values to save for actions
            tgt_q_values = rewards + self.gamma * \
                self.model_target.predict(next_states)[np.arange(len(next_states)), 
                                            selected_actions]
        else:
            # Outputs a vector tgt_q_values to save for actions
            tgt_q_values = rewards + self.gamma * \
                    np.amax(self.model_target.predict(next_states), axis=1)

        # Update q_values_mat according to computed target values
        tgt_q_values[done_flags==True] = rewards[done_flags==True]
        batch_q_values_arr[np.arange(len(states)), actions] = tgt_q_values

        fit_data=self.model.fit(states, batch_q_values_arr,
                batch_size= self.minibatch_size, epochs=1, verbose=0)
#        print(fit_data.history['loss'])
        if update_tgt:
            # Update the target model to the current model
            self.model_target.set_weights(self.model.get_weights())
#        print("updated")
        return fit_data.history['loss']

    def minibatch_update(self, replay_batch_states, update_tgt=False):
        batch_states = []
        batch_q_values =[]
        # Gather target q_values from replay memory states
        for state, action, reward, next_state, done in replay_batch_states:
            # q_values for actions not taken will equal predicted action values
            ## thereby only loss of action taken is used in error term 
            q_values = self.model.predict(state)[0] 
            # Use target model for estimate of q(s',a')
            tgt_q_value = None
            if self.is_DDQN:
                act = np.argmax(self.model.predict(next_state)[0])
                tgt_q_value = reward + self.gamma * self.model_target.model.predict(next_state)[0][act]
            else:
                tgt_q_value = reward + self.gamma * np.amax(self.model_target.model.predict(next_state)[0])
            if done:
                q_values[action] = reward
            else:
                q_values[action] = tgt_q_value
            batch_states.append(state[0])
            batch_q_values.append(q_values)

        return np.array(batch_states), np.array(batch_q_values)

        self.model.fit(np.array(batch_states), np.array(batch_q_values),
                batch_size= self.minibatch_size, epochs=1, verbose=0)

        if update_tgt:
            # Update the target model to the current model
            self.model_target.set_weights(self.model.get_weights())

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        print ("Burning in memory ...", self.memory.burn_in, "samples to collect.")
        state = self.env.reset()
        state =  trans_state(state)
#        state = np.expand_dims(state,0)
        for i in range(self.memory.burn_in):
#            self.env.render()
            action = self.env.action_space.sample()
            next_state, reward1, done, info = self.env.step(action)
            next_state =  trans_state(next_state)
            burned_state = np.concatenate((state, [action], next_state))
            burned_state = burned_state.ravel()
            burned_state = np.expand_dims(burned_state,0)
            state = next_state
            action = self.env.action_space.sample()
            next_state, reward2, done, info = self.env.step(action)
            next_state =  trans_state(next_state)            
            burned_next_state = np.concatenate((state, [action], next_state))
            burned_next_state = burned_next_state.ravel()
            burned_next_state = np.expand_dims(burned_next_state,0)
            self.memory.append((burned_state, action, reward2, burned_next_state, done))
            burned_state = burned_next_state
            state = next_state
        print ("Burn-in complete.")

    def step_and_update(self, state, step_number):
        # Returns 2-tuple of:
        ## (action, (next_state, reward, done, info) )
        ## action is chosen action of eps greedy policy for dqn
        ## (next_state, reward, done, info) is output of env.step(action)     
        q_values = self.model.predict(state)[0]
        action =  self.epsilon_greedy_policy(q_values)
        next_state, reward, done, info = self.env.step(action)
        next_state =  trans_state(next_state)

        # Do a batch update after kth action taken
        update_tgt_flag = False
        update_tgt_flag = not self.is_DDQN and step_number % self.n_steps_before_update_tgt == 0
        if step_number % self.k_steps_before_minibatch == 0:
            minibatch = self.memory.sample_batch(self.minibatch_size)
            lss=self.fast_minibatch_update(minibatch, update_tgt=update_tgt_flag)

            # If DDQN, then we need to randomly swap the two QNetworks
            if self.is_DDQN: ## added by Ibrahim Fri 1:23AM 
                shuffled_models = [self.model, self.model_target]
                np.random.shuffle(shuffled_models)
                self.model, self.model_target = shuffled_models
    
        return action, (next_state, reward, done, info), lss

    def train(self, n_steps_to_reset=1000000000000000):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        eps_counter = 1
        episodes_return = []
        self.avg_training_episodes_return = []
        self.avg_performance_episodes_return = []

        filename = 'weights_'+str(self.env_name)+'_'+str(self.model_name)+'_0'
        print(filename)
        self.model.save_model_weights(filename)
        # Step-wise vars 
        n_steps = 0 
        state = self.env.reset()
        state = trans_state(state)
        action = self.env.action_space.sample()
        next_state, reward1, done, info = self.env.step(action)
        next_state =  trans_state(next_state)
        sas_state = np.concatenate((state, [action], next_state), axis=0)
        sas_state = sas_state.ravel()
        sas_state = np.expand_dims(sas_state,0)
#        state = np.expand_dims(state,0)
        returns = 0
        self.step_rewards =[]
        # Run num_episodes many episodes
        for episode in range(self.num_episodes):
#            self.env.render()
#            done = False
            action, step_info, lss = self.step_and_update(sas_state, n_steps)
            next_state1, reward, done, _ = step_info
            n_steps += 1
            state = next_state
            next_state = next_state1
            next_sas_state = np.concatenate((state, [action], next_state), axis=0)
            next_sas_state = next_sas_state.ravel()
            next_sas_state = np.expand_dims(next_sas_state,0)
#            next_state1 = np.expand_dims(next_state1,0)
#            if episode==0: print(sas_state, action, reward, next_sas_state, done)
            self.memory.append((sas_state, action, reward, next_sas_state, done))
            sas_state = next_sas_state
            
            returns += reward   
            self.step_rewards.append(reward)
#            while not done:
#                action, step_info = self.step_and_update(state, n_steps)
#                next_state, reward, done, _ = step_info
#                n_steps += 1
#                next_state = np.expand_dims(next_state,0)
#                self.memory.append((state, action, reward, next_state, done))
#                state = next_state
#                returns += reward 

#            episodes_return.append(returns)
            print("Episode: ", eps_counter," Reward: ", reward, "Epsilon:", round(self.epsilon,4),\
                    "LR:", keras.backend.eval(self.model.model.optimizer.lr), "loss:", round(lss[0],4))

            ## get the points of the training curve
#            if eps_counter % self.num_of_episodes_to_update_train_and_perf_curve == 0:
##                self.avg_training_episodes_return.append(np.mean(episodes_return))
#                self.avg_training_episodes_return.append(returns)
##                episodes_return = []
#                returns = 0
            if eps_counter % save_weights_num_episodes == 0:
                filename = 'weights_'+str(self.env_name)+'_'+str(self.model_name)+'_'+str(eps_counter)
                print(filename)
                self.model.save_model_weights(filename)
            if eps_counter % n_steps_to_reset == 0:
                state = self.env.reset()
                state = trans_state(state)
                action = self.env.action_space.sample()
                next_state, reward1, done, info = self.env.step(action)
                next_state =  trans_state(next_state)
                sas_state = np.concatenate((state, [action], next_state), axis=0)
                sas_state = sas_state.ravel()
                sas_state = np.expand_dims(sas_state,0)
                episodes_return.append(returns)
                returns = 0
            eps_counter += 1
#        plt.figure()
##        plt.plot(self.avg_training_episodes_return,label='training_curve')
##        plt.plot(episodes_return,label='training_curve')
#        plt.plot(self.step_rewards,label='training_curve')
##        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(n_steps_to_reset) + ' steps)')
#        plt.xlabel('Training steps')
#        plt.ylabel('Average Reward per Episode')
#        plt.legend(loc='best')
#        plt.show()
        return self.step_rewards

    def test_stats(self, n_test_episodes,n_steps, model_file=None):
        # Returns the average reward over n_test_episodes of trained model
        ## and standard deviation
        total_returns = []
        for ep in range(n_test_episodes):
            state = self.env.reset()
            state = trans_state(state)
            action = self.env.action_space.sample()
            next_state, reward1, done, info = self.env.step(action)
            next_state =  trans_state(next_state)
            sas_state = np.concatenate((state, [action], next_state), axis=0)
            sas_state = sas_state.ravel()
            sas_state = np.expand_dims(sas_state,0)
            total_return = 0
            for step in range(n_steps):
                q_values = self.model.predict(sas_state)[0]
                action = self.epsilon_greedy_policy( q_values, force_epsilon=0.05)
                next_state1, reward, done, info = self.env.step(action)
                next_state1 =  trans_state(next_state1)
                state = next_state
                next_state = next_state1
                next_sas_state = np.concatenate((state, [action], next_state), axis=0)
                next_sas_state = next_sas_state.ravel()
                next_sas_state = np.expand_dims(next_sas_state,0)
                self.memory.append((sas_state, action, reward, next_sas_state, done))
                sas_state = next_sas_state
                total_return += reward 
            total_returns.append( total_return )

        # Compute test stats    
        average_test_return = np.mean(total_returns)
        std_test_return = np.std(total_returns)
        print("Mean total reward over %s episodes: %s +/- %s" % \
                (n_test_episodes,  average_test_return, std_test_return))
        return average_test_return, std_test_return
    
    def get_rewards(self, n_steps, model_file=None):
        # Returns the average reward over n_test_episodes of trained model
        ## and standard deviation
        rewards=[]
        state = self.env.reset()
        state = trans_state(state)
        action = self.env.action_space.sample()
        next_state, reward1, done, info = self.env.step(action)
        next_state =  trans_state(next_state)
        sas_state = np.concatenate((state, [action], next_state), axis=0)
        sas_state = sas_state.ravel()
        sas_state = np.expand_dims(sas_state,0)
        for step in range(n_steps):
            q_values = self.model.predict(sas_state)[0]
            action = self.epsilon_greedy_policy( q_values, force_epsilon=0.05)
            next_state1, reward, done, info = self.env.step(action)
            next_state1 =  trans_state(next_state1)
            state = next_state
            next_state = next_state1
            next_sas_state = np.concatenate((state, [action], next_state), axis=0)
            next_sas_state = next_sas_state.ravel()
            next_sas_state = np.expand_dims(next_sas_state,0)
            self.memory.append((sas_state, action, reward, next_sas_state, done))
            sas_state = next_sas_state
            rewards.append(reward) 
        return rewards
    
    def performance_plot_data(self,num_episodes,n_steps, model_file=None):
        total_returns = []
        for ep in range(num_episodes):
            state = self.env.reset()
            state = trans_state(state)
            action = self.env.action_space.sample()
            next_state, reward1, done, info = self.env.step(action)
            next_state =  trans_state(next_state)
            sas_state = np.concatenate((state, [action], next_state), axis=0)
            sas_state = sas_state.ravel()
            sas_state = np.expand_dims(sas_state,0)
            total_return = 0
            for step in range(n_steps):
                q_values = self.model.predict(sas_state)[0]
                action = self.epsilon_greedy_policy( q_values, force_epsilon=0.05)
                next_state1, reward, done, info = self.env.step(action)
                next_state1 =  trans_state(next_state1)
                state = next_state
                next_state = next_state1
                next_sas_state = np.concatenate((state, [action], next_state), axis=0)
                next_sas_state = next_sas_state.ravel()
                next_sas_state = np.expand_dims(next_sas_state,0)
                self.memory.append((sas_state, action, reward, next_sas_state, done))
                sas_state = next_sas_state
                total_return += reward 
            total_returns.append( total_return )
        return (sum(total_returns)/num_episodes)

    
    def performance_curves_from_weight_files(self, num_resets, n_steps):
        self.avg_performance_episodes_return = []
#        self.avg_performance_episodes_return_2SLA = []
        start = 0
        stop = self.num_episodes +1
        step = self.num_of_episodes_to_update_train_and_perf_curve
        num_episodes_for_performance_curve = num_resets
        print("Printing performance curve plot...")
        for indx in range(start, stop,step):
            filename = 'weights_'+str(self.env_name)+'_'+str(self.model_name)+'_'+str(indx)
            print("loading weight file: ",filename)            
            self.model.load_model_weights(filename)
            self.avg_performance_episodes_return.append(self.performance_plot_data(num_episodes_for_performance_curve, n_steps))
#            self.avg_performance_episodes_return_2SLA.append(self.performance_plot_data_2_steps_LA(num_episodes_for_performance_curve))
            

    def plots(self):
        plt.figure()
        plt.plot(self.avg_performance_episodes_return,label='performance_curve')
#        plt.plot(self.avg_performance_episodes_return_2SLA,label='performance_curve 2 steps look_ahead')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(self.num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.legend(loc='best')
        plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='NEL-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_name',type=str, default="dqn")
    return parser.parse_args()

if __name__ == '__main__':

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # Gather commandline args
    args = parse_arguments()
    environment_name = args.env
    model_name = args.model_name

    agent = Deep_Agent(environment_name, model_name, num_episodes=num_episodes, curve_episodes=1000)
    agent.burn_in_memory()
    training_step_rewards = agent.train(n_steps_to_reset=100000000000000000)
    plot("tsr",training_step_rewards)
    rewards = agent.get_rewards(10000)
    plot("pr",rewards)
    filename = "rewards_file"
    with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([training_step_rewards,rewards], f,protocol=2)    
#    u, std = agent.test_stats(10,100)  # 6.e
#    agent.performance_curves_from_weight_files(10, 100)
#    agent.plots()

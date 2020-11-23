import sys
import argparse
import numpy as np
import keras
import greedy_policy
import pickle
import random
import gym
import timeit
import nel
import canonical_plot
from keras.utils.np_utils import to_categorical

# some of the code here was borrowed from https://github.com/agakshat#

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

class Imitation():
    def __init__(self, args, model_config_path, expert_weights_path):

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(512, input_dim=367, name='dense_1', activation='relu', use_bias=True, kernel_initializer='uniform'))
        self.model.add(keras.layers.Dense(64, name='dense_2', activation='relu', use_bias=True, kernel_initializer='uniform'))
        self.model.add(keras.layers.Dense(18, name='dense_3', activation='relu', use_bias=True, kernel_initializer='uniform'))
        self.model.add(keras.layers.Dense(3, name='dense_4', activation='softmax', use_bias=True, kernel_initializer='uniform'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.args = args

    def run_expert(self, env, render=False):
        # Generates an episode by running the expert policy on the given env.
        return Imitation.generate_episode_expert(self.args, env, render)

    def run_model(self, env, render=False):
        # Generates an episode by running the cloned policy on the given env.
        return Imitation.generate_episode_model(self.args, self.model, env, render)

    @staticmethod
    def generate_episode_expert(args, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        #Run an episode using the model
        obs = env.reset()
        done = False
        step = 0
        while not done and step <= args.max_steps:
            if args.render:
                env.render()
            states.append(trans_state(obs))
#            obs = obs.reshape(1, obs.shape[0])
#            next_act_probs = model.predict(obs)
#            next_act = np.argmax(next_act_probs)    #Pick most likely action
            next_act = policy_fn(env, obs['vision'])
            categorical_labels = to_categorical(next_act, num_classes=env.action_space.n)   #COnverts to one hot outputs
            actions.append(categorical_labels)          #For expert action chosen is stored
            next_obs, reward, done, _ = env.step(next_act)
            rewards.append(reward)
            obs = next_obs
            step += 1

        return np.array(states), np.array(actions), np.array(rewards)
    
    @staticmethod
    def save_episode_expert(args, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        print("Saving training data by running expert policy")
        states = []
        actions = []
        rewards = []

        #Run an episode using the model
        obs = env.reset()
        done = False
        step = 0
        while not done and step <= args.max_steps:
            if args.render:
                env.render()
            states.append(trans_state(obs))
#            obs = obs.reshape(1, obs.shape[0])
#            next_act_probs = model.predict(obs)
#            next_act = np.argmax(next_act_probs)    #Pick most likely action
            next_act = policy_fn(env, obs['vision'])
            categorical_labels = to_categorical(next_act, num_classes=env.action_space.n)   #COnverts to one hot outputs
            actions.append(categorical_labels)          #For expert action chosen is stored
            next_obs, reward, done, _ = env.step(next_act)
            rewards.append(reward)
            obs = next_obs
            step += 1
            if step % 100 ==0:
                print("step=",step)
        filename = 'expert_states_actions_rewards_' + str(args.max_steps)
        with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([ np.array(states), np.array(actions),np.array(rewards)], f,protocol=2)

    @staticmethod
    def generate_episode_model(args, model, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []

        #Run an episode using the model
        obs = env.reset()
        done = False
        step = 0
        while not done and step <= args.max_steps:
            if args.render:
                env.render()
            obs = trans_state(obs)
            states.append(obs)
            obs = obs.reshape(1, obs.shape[0])
            next_act_probs = model.predict(obs)
            next_act = np.argmax(next_act_probs)    #Pick most likely action
            actions.append(next_act)          #For model probabilities need to be stored
            next_obs, reward, done, _ = env.step(next_act)
            rewards.append(reward)
            obs = next_obs
            step += 1

        return states, actions, rewards
    
    def train(self, env, pickle_flag, num_episodes=10, num_epochs=50, render=False):
        # Trains the model on training data generated by the expert policy.
        # Args:
        # - env: The environment to run the expert policy on. 
        # - num_episodes: # episodes to be generated by the expert.
        # - num_epochs: # epochs to train on the data generated by the expert.
        # - render: Whether to render the environment.
        # Returns the final loss and accuracy.
        # TODO: Implement this method. It may be helpful to call the class
        #       method run_expert() to generate training data.

        args = self.args


        for i in range(num_episodes):
            if pickle_flag: 
                file_name = 'expert_states_actions_rewards_'+str(args.max_steps)
                with open(str(file_name)+".pkl", 'rb') as f:  # Python 3: open(..., 'rb')
                    states, actions, rewards  = pickle.load(f) 
            else:
                #Construct the training dataset
                print("Generating training data by running expert policy")
                states, actions, rewards = self.run_expert(env)
            if i == 0:
                train_states = states
                train_actions = actions
                train_rewards = rewards
            else:
                train_states = np.concatenate((train_states, states), axis=0)        
                train_actions = np.concatenate((train_actions, actions), axis=0)
                train_rewards = np.concatenate((train_rewards, rewards), axis=0) 

        print("Train data has been generated using the expert policy")

        #Train the model using data from the expert
        self.model.fit(train_states, train_actions, epochs=num_epochs, batch_size=args.batch_size)

        #Evaluate the model
        # evaluate the model
        scores = self.model.evaluate(train_states, train_actions)
        #print("\n%s: %.2f%%" % (imitater.model.metrics_names[1], scores[1]*100))

        loss = scores[0]
        acc = scores[1]*100
        self.model.save_weights("imitater_" + str(num_episodes)+".h5")

        return loss, acc

    def test(self, env,  num_of_episodes = 1):   
        trained_weights_path = "imitater_" + str(self.args.episodes)+".h5"
        tot_rewards = []
        self.model.load_weights(trained_weights_path)
        for i in range(num_of_episodes):
            _, _, rewards = self.run_model(env) 
            tot_rewards.append(rewards)
        print("saving performance rewards file")
        filename = 'pr_rewards_imitation_learning'
        with open(str(filename)+".pkl", 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(rewards, f,protocol=2)
        return np.mean(tot_rewards), np.std(tot_rewards)

    def test_expert(self, env,  num_of_episodes = 100):   
        tot_rewards = []
        for i in range(num_of_episodes):
            _, _, rewards = self.run_expert(env) 
            tot_rewards.append(np.sum(rewards))

        return np.mean(tot_rewards), np.std(tot_rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--expert-weights-path', dest='expert_weights_path',
                        type=str, default='LunarLander-v2-weights.h5',
                        help="Path to the expert weights file.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)
    
    parser_group.add_argument('--test', dest='test',
                              action='store_true',
                              help="Whether to run in test mode")
    parser_group.add_argument('--train', dest='test',
                              action='store_false',
                              help="Whether to run in train mode")
    parser.set_defaults(test=False)
    
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
    parser.add_argument('--episodes', type=int, default=1, metavar='N',
                    help='Number of episodes to generate from expert')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='Batch Size')
    parser.add_argument('--max_steps', type=int, default=10000, metavar='N',
                    help='Maximum length of running an episode')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of epochs to train the model for')
    return parser.parse_args()



# Parse command-line arguments.
args = parse_arguments()
model_config_path = args.model_config_path
expert_weights_path = args.expert_weights_path
render = args.render

# Create the environment.
env = gym.make('NEL-v0')
pickle_flag = True
imitater = Imitation(args, args.model_config_path, args.expert_weights_path)
if not args.test:
    start_time = timeit.default_timer()
    imitater.save_episode_expert(args, env)
    elapsed_time = timeit.default_timer() - start_time
    print("Time="+str(elapsed_time))
    loss, acc = imitater.train(env, pickle_flag, args.episodes, args.epochs, args.render)
#    #imitater.save_weights("imitater_" + str(args.episodes)+".h5")
    print("The imitater model has been trained")
    print("Accuracy is: "+str(acc) + " and Average Train loss is: "+str(loss))

else:
    mean, std = imitater.test(env)
    print("Number of training episodes: " +str(args.episodes) + " \nMean reward is: " + str(mean) + " and Std of rewards is: "+str(std)) 
    #mean, std = imitater.test_expert(env)
    #print("Testing Expert Model: Mean reward is: " + str(mean) + " and Std of rewards is: "+str(std))


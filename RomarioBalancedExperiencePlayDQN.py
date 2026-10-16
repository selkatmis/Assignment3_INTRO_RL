# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import gymnasium as gym
from collections import deque
from collections import defaultdict
import matplotlib.pyplot as plt
from test_script import QNetwork
from test_script import bar_plot, test_pole_length, test_script

class ReplayBuffer:
    """
    The storage for transitions. It can add new transitions, with a limited capacity.
    You can fetch a sample batch or the buffer's length.
    """
    # Initialisatin and setting a limit on how many transitions the buffer can store
    def __init__(self, capacity):
        good_buffer_capacity = capacity//2
        bad_buffer_capacity = capacity - good_buffer_capacity
        
        #deque is a double ended queue
        self.good_episode_buffer = deque(maxlen=good_buffer_capacity) #buffer for good epsiodes
        self.bad_episodes_buffer = deque(maxlen=bad_buffer_capacity) #buffer for bad epsiodes
        self.current_episode_transitions = [] #list to keep track of the transition of the current episode

    # Function to add a new transition into the list that keepa track of transitions in the episode
    def add_transition(self, state,action,reward,next_state,terminated):
        transition: tuple = (state,action,reward,next_state,terminated)
        self.current_episode_transitions.append(transition)

    #function to classify each episode and add their respective transitions to the correct buffer. 
    def episode_classification(self, total_episode_rewards, good_episode_threshold):
        if total_episode_rewards >= good_episode_threshold:
            self.good_episode_buffer.extend(self.current_episode_transitions)
        elif total_episode_rewards < good_episode_threshold:
            self.bad_episodes_buffer.extend(self.current_episode_transitions)
        self.current_episode_transitions = []

    # Samples an x/2 amount of transitions from both buffers
    def sample_batch(self, batch_size):
        good_episodes_transition_amount = batch_size//2
        bad_episodes_transition_amount = batch_size - good_episodes_transition_amount
        if len(list(self.bad_episodes_buffer)) == 0:
            batch = random.sample(list(self.good_episode_buffer), batch_size)
            random.shuffle(batch)
            return batch

        if len(list(self.good_episode_buffer)) == 0:
            batch = random.sample(list(self.bad_episodes_buffer), batch_size)
            random.shuffle(batch)
            return batch
        
        elif bad_episodes_transition_amount <= len(list(self.bad_episodes_buffer)):
            batch = random.sample(list(self.bad_episodes_buffer), bad_episodes_transition_amount)
            if good_episodes_transition_amount <= len(list(self.good_episode_buffer)):
                                batch += random.sample(list(self.good_episode_buffer), good_episodes_transition_amount)
            else:
                batch += random.choices(list(self.good_episode_buffer), k=good_episodes_transition_amount)

        else:
            batch = random.choices(list(self.bad_episodes_buffer), k=bad_episodes_transition_amount)

            if good_episodes_transition_amount <= len(list(self.good_episode_buffer)):
                        batch += random.sample(list(self.good_episode_buffer), good_episodes_transition_amount)
            else:
                batch += random.choices(list(self.good_episode_buffer), k=good_episodes_transition_amount)

        random.shuffle(batch)
        return batch

    # Returns the amount of transitions in the buffer
    def __len__(self):
        return len(self.good_episode_buffer) + len(self.bad_episodes_buffer)


def select_action(state,evaluation_network,epsilon):
    """Select an action using an epsilon-greedy policy.

    - With the chance of epsilon, choose a random action
    - Otherwise, select the action with the highest Q-value

    Args:
        state: Current CartPole state containing four observations.
        evaluation_network: Q-network used to predict action values.
        epsilon: Probability of selecting a random action.

    Returns:
        The selected action (0 for left or 1 for right).
    """

    random_number = random.random() # between 0 and 1

    if random_number < epsilon:
        #Explore
        action = random.randrange(2)
    else: 
        #Exploit
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # No gradients needed to choose an action, as we're only choosing an action
        with torch.no_grad():
            Q_values = evaluation_network(state_tensor)
            action = Q_values.argmax().item() # Choose highest Q-value

    return action

def main(epsilon,gamma,alpha,Q_network,
        episodes,pole_lengths,batch_size,
        buffer_size,update_target_network,
        epsilon_decay,epsilon_minimum=0.1):

    """Train a DQN agent on CartPole with varying pole lengths.
    
    The agent uses an epsilon-greedy policy in select_action() for picking actions. The evaluation network 
    learns through backpropagation, while the target network provides Q-values from an earlier 
    copy of the evaluation network to calculate losses. Pole lengths are randomly selected
    at the beginning of an episode. ReplayBuffer is a storage class, storing the 
    deque containing the state transitions.

    Args:
        epsilon: Initial probability of choosing a random action.
        gamma: Discount factor for future rewards.
        alpha: Learning rate used by the Adam optimizer.
        Q_network: Usage of an existing Q-network. If there is not 
        one provided, a new Q-network will be initialised.
        episodes: Number of training episodes.
        pole_lengths: Collection of pole lengths to sample during
            training. If None, use 30 lengths from 0.4 to 1.8.
        batch_size: Number of transitions sampled per learning update.
        buffer_size: Maximum number of transitions in the replay buffer.
        update_target_network: Number of episodes between target
            network weight updates.
        epsilon_decay: Factor used to decrease epsilon, which is applied
            at the beginning of each episode.
        epsilon_minimum: Minimum exploration probability.
            Defaults to 0.1.
        good_episode_threshold: threshold for how much total rewards a episode must have to
            be clasified as good.
    
    Returns:
        A tuple containing:
            evaluation_network: The trained Q-network.
            target_network: The final target network.
            episode_returns: Total reward obtained in each training episode.
    """

    # Initialisation of the environment
    env = gym.make("CartPole-v1")

    # Observation: 4 values
    # Action: 2 values
    if Q_network == None:
        evaluation_network = QNetwork(4,2)
    else: 
        evaluation_network = Q_network

    target_network = QNetwork(4,2)
    target_network.load_state_dict(evaluation_network.state_dict())
    target_network.eval()


    # Initialisation of the optimizer
    optimizer = optim.Adam(evaluation_network.parameters(), lr=alpha)

    # Initialisation of replay buffer
    replay_buffer = ReplayBuffer(buffer_size)

    # Default pole lengths for training if one is not provided
    if pole_lengths == None:
        pole_lengths = np.linspace(0.4, 1.8, 30)

    # Training statistics
    episode_returns = []
    training_records = []
    total_env_steps = 0
    gradient_updates = 0

    # A copy of the original provided epsilon
    epsilon_copy = epsilon

    for episode in range(episodes):
        pole_length = np.random.choice(pole_lengths)
        env.unwrapped.length = pole_length

        #environment reset
        state = env.reset()[0]
        episode_reward = 0.0

        if epsilon_copy > epsilon_minimum:
            epsilon_copy = max(epsilon_minimum, epsilon_copy * epsilon_decay)

        finished = False

        while not finished:
            action = select_action(state,evaluation_network,epsilon_copy)

            next_state, reward, terminated, truncated, _ = env.step(action)

            total_env_steps += 1

            finished = terminated or truncated

            replay_buffer.add_transition(
                state, action, reward, next_state, terminated)

            # If the replay buffer has enough transitions, start
            # updating the evaluation network
            if len(replay_buffer) >= batch_size:
                # Sampling a batch from replay buffer
                batch = replay_buffer.sample_batch(batch_size)

                # Instead of providing separate tuples for zip(), we use *batch, 
                # which zips each index of the transition into separate lists 
                states, actions, rewards, next_states, dones = zip(*batch)

                # Converting these same lists into tensors
                states_t = torch.FloatTensor(np.stack(states))
                actions_t = torch.LongTensor(actions).unsqueeze(1)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t = torch.FloatTensor(np.stack(next_states))
                dones_t = torch.FloatTensor(dones).unsqueeze(1)
                
                # Predict Q-values and select the value of each action taken
                current_q = evaluation_network(states_t).gather(1,actions_t)

                # Target values
                # Without the need of gradients as we do not update the target network here
                with torch.no_grad():
                    next_max = target_network(next_states_t).max(1)[0].unsqueeze(1)
                    td_target = rewards_t + gamma * next_max * (1 - dones_t)
            
                # Calculation of the loss
                loss = nn.MSELoss()(current_q, td_target)

                # Backpropagation and Optimization + Gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(evaluation_network.parameters(), 1.0)
                optimizer.step()
                gradient_updates += 1

            episode_reward += reward
            state = next_state

        #dynamically change the threshold of good episodes
        if len(episode_returns) > 1:
             good_episode_threshold = np.mean(episode_returns[-100:])
        else:
             good_episode_threshold = episode_reward
        #classify the episode (good or bad)
        replay_buffer.episode_classification(episode_reward, good_episode_threshold)
        
        # Updating the target network after x amounts of episodes, by copying 
        # the evaluation network on target network
        if episode % update_target_network == 0:
            target_network.load_state_dict(evaluation_network.state_dict())

        episode_returns.append(episode_reward)

        # Storing the training statistics
        training_records.append((
            episode, episode_reward, total_env_steps,
            gradient_updates, epsilon_copy
        ))
                    
        # Printing progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Epsilon: {epsilon_copy:.3f}")

    env.close()
    
    torch.save(evaluation_network.state_dict(), "weights/Romario_Balanced_replay.pth")
    # Storing the data needed for evaluation
    with open("results/data/Romario_Balanced_replay.csv", "w",
            newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "total_env_steps",
                        "gradient_updates", "epsilon"])
        writer.writerows(training_records)

        print(r"C:\Users\roanc\Downloads\RL-project\Romario_Balanced_replay.csv")

    return (evaluation_network, target_network, episode_returns)

trained_network, target_network, rewards = main(
    epsilon=1.0,
    gamma=0.99,
    alpha=0.001,
    Q_network=None,
    episodes=1500,
    pole_lengths=None,
    batch_size=64,
    buffer_size=50000,
    update_target_network=10,
    epsilon_decay=0.995
)

test_script()

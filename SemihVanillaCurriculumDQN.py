import csv
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from test_script import QNetwork


class ReplayBuffer:
    """
    The storage for transitions. It can add new transitions, with a limited capacity.
    You can fetch a sample batch or the buffer's length.
    """
    # Initialisatin and setting a limit on how many transitions the buffer can store
    def __init__(self, capacity):
        self.buffer =  deque(maxlen=capacity) #deque is a double ended queue

    # Function to add a new transition into the buffer
    def add_transition(self, state,action,reward,next_state,terminated):
        transition: tuple = (state,action,reward,next_state,terminated)
        self.buffer.append(transition)

    # Samples an x amount of transitions from the buffer
    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    # Returns the amount of transitions in the buffer
    def __len__(self):
        return len(self.buffer)


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
        epsilon_decay,epsilon_minimum=0.1,
        curriculum_strength=0.8, warmup_rounds=2, history_size=10):

    """Train DQN with performance-based adaptive pole-length selection.
    
    The agent uses an epsilon-greedy policy in select_action() for picking actions. The evaluation network 
    learns through backpropagation, while the target network provides Q-values from an earlier
    copy of the evaluation network to calculate losses. Pole lengths are selected
    based on the agent's recent performance after an initial warm-up period.
    ReplayBuffer stores and samples transitions as in vanilla DQN.

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
        curriculum_strength: How strongly sampling favors low-reward pole lengths.
            Defaults to 0.8, leaving 20% of sampling probability uniform.
        warmup_rounds: Number of initial passes through all pole lengths.
            Defaults to 2.
        history_size: Number of recent rewards recorded for each pole length.
            Defaults to 10.
    
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
    if Q_network is None:
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
    if pole_lengths is None:
        pole_lengths = np.linspace(0.4, 1.8, 30)
    pole_lengths = np.asarray(pole_lengths, dtype=float)

    # To get an estimatate of the performance, for each pole length, 
    # we keep the last few episode rewards  
    length_rewards = [deque(maxlen=history_size) for _ in pole_lengths]

    # We train on every pole length couple times before we start adapting (warm-up)
    warmup_indices = list(range(len(pole_lengths))) * warmup_rounds
    random.shuffle(warmup_indices)

    # Training statistics
    episode_returns = []
    training_records = []
    total_env_steps = 0
    gradient_updates = 0

    # A copy of the original provided epsilon
    epsilon_copy = epsilon

    for episode in range(episodes):
        # Warm-up first
        if episode < len(warmup_indices):
            length_index = warmup_indices[episode]
        else:
            # We calculate the mean for each pole length's last rewards
            reward_means = np.array([np.mean(rewards) for rewards in length_rewards])

            # Then we calculate the mean reward across all pole lengths
            all_rewards_mean = np.mean(reward_means)

            # We then select the pole lengths that whos mean reward is below the overall mean 
            difficult_lengths = np.where(reward_means < all_rewards_mean)[0]

            # We prioritise difficult lengths, but sometimes we sample from all lengths
            if len(difficult_lengths) > 0 and random.random() < curriculum_strength:
                length_index = np.random.choice(difficult_lengths)
            else:
                length_index = np.random.choice(len(pole_lengths))

        pole_length = float(pole_lengths[length_index])
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

        # Updating the target network after x amounts of episodes, by copying 
        # the evaluation network on target network
        if episode % update_target_network == 0:
            target_network.load_state_dict(evaluation_network.state_dict())

        episode_returns.append(episode_reward)

        # Remember how the DQN performed at the pole length used in this episode
        length_rewards[length_index].append(episode_reward)

        # Storing the training statistics
        training_records.append((
            episode, episode_reward, total_env_steps,
            gradient_updates, epsilon_copy, pole_length
        ))
                    
        # Printing progress
        if episode % 50 == 0:
            avg_reward = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
            print(f"Episode {episode}/{episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Epsilon: {epsilon_copy:.3f}")
    
    env.close()

    # Saving the training
    torch.save(evaluation_network.state_dict(), "weights/semih_adaptive_curriculum_dqn.pth")

    with open("results/data/semih_vanilla_curriculum_training.csv", "w",
              newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "total_env_steps",
                         "gradient_updates", "epsilon", "pole_length"])
        writer.writerows(training_records)

    print("Saved: weights/semih_adaptive_curriculum_dqn.pth")
    print("Saved: results/data/semih_adaptive_curriculum_training.csv")

    return (evaluation_network, target_network, episode_returns)


if __name__ == "__main__":
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
        epsilon_decay=0.995,
        epsilon_minimum=0.1,
        curriculum_strength=0.8,
        warmup_rounds=2,
        history_size=10
    )

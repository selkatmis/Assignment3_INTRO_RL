import csv
import multiprocessing as mp
import os
import random
from collections import deque
from queue import Empty

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


def snapshot(network):
    """Making an independent copy of the network weights for the actor."""
    return {
        name: value.detach().cpu().numpy().copy()
        for name, value in network.state_dict().items()
    }


def load_snapshot(network, weights):
    """Load copied network weights received from the learner."""
    network.load_state_dict({
        name: torch.from_numpy(value)
        for name, value in weights.items()
    })


def actor_worker(experience_queue, weights_queue, initial_weights,
                 episodes, pole_lengths, epsilon, epsilon_decay,
                 epsilon_minimum):
    """Play CartPole in a separate process and send transitions to main().

    The actor only chooses actions and collects experiences. It does not
    calculate TD targets or perform backpropagation.

    Args:
        experience_queue: Queue for sending transitions to the learner.
        weights_queue: Queue for receiving updated evaluation-network weights.
        initial_weights: Copy of the learner's initial network parameters.
        episodes: Number of episodes for this actor to play.
        pole_lengths: Pole lengths available during training.
        epsilon: Starting exploration probability.
        epsilon_decay: Epsilon decay factor per episode.
        epsilon_minimum: Minimum exploration probability.
    """

    # The actor has its own network for selecting actions
    actor_network = QNetwork(4, 2)
    load_snapshot(actor_network, initial_weights)
    actor_network.eval()

    epsilon_copy = epsilon
    env = None

    try:
        env = gym.make("CartPole-v1")

        for episode in range(episodes):
            # Check for newer weights from the learner before each episode
            try:
                while True:
                    new_weights = weights_queue.get_nowait()
                    load_snapshot(actor_network, new_weights)
            except Empty:
                pass  # There are no more updated weights waiting

            # Select a random pole length for this episode
            pole_length = np.random.choice(pole_lengths)
            env.unwrapped.length = float(pole_length)

            state = env.reset()[0]
            episode_reward = 0.0

            if epsilon_copy > epsilon_minimum:
                epsilon_copy = max(epsilon_minimum, epsilon_copy * epsilon_decay)

            finished = False
            chunk = []

            while not finished:
                # This is the same epsilon-greedy function as in vanilla DQN
                action = select_action(state, actor_network, epsilon_copy)

                next_state, reward, terminated, truncated, _ = env.step(action)
                finished = terminated or truncated

                # The actor cannot directly add to the learner's replay buffer,
                # so it collects transitions and sends them through a queue
                chunk.append((state.copy(), action, reward,
                              next_state.copy(), terminated))

                episode_reward += reward
                state = next_state

                # Send transitions in small chunks instead of waiting
                # for the entire episode to finish
                if len(chunk) >= 32:
                    experience_queue.put(("transitions", chunk))
                    chunk = []

            # Send any transitions that remain from this episode
            if chunk:
                experience_queue.put(("transitions", chunk))

            # Tell the learner that this episode has ended
            experience_queue.put(("episode", episode, episode_reward, epsilon_copy))

    except Exception as error:
        # Report actor errors to the main process
        experience_queue.put(("error", repr(error)))
    finally:
        if env is not None:
            env.close()
        experience_queue.put(("done",))


def main(epsilon, gamma, alpha, Q_network,
         episodes, pole_lengths, batch_size,
         buffer_size, update_target_network,
         epsilon_decay, epsilon_minimum=0.1):
    """Train asynchronous DQN on CartPole with varying pole lengths.

    The actor_worker() process uses an epsilon-greedy policy to play CartPole
    and sends transitions to main() through experience_queue. The learner
    stores these transitions in ReplayBuffer and updates the evaluation
    network through backpropagation, while the actor continues playing.
    The target network provides Q-values from an earlier copy of the
    evaluation network to calculate TD targets.

    Args:
        epsilon: Initial probability of choosing a random action.
        gamma: Discount factor for future rewards.
        alpha: Learning rate used by the Adam optimizer.
        Q_network: Existing Q-network, or None to initialise a new one.
        episodes: Number of training episodes.
        pole_lengths: Lengths to sample, or None for 30 lengths from 0.4 to 1.8.
        batch_size: Number of transitions sampled for each learning update.
        buffer_size: Maximum number of transitions stored in replay buffer.
        update_target_network: Number of episodes between network synchronisations.
        epsilon_decay: FFactor used to decrease epsilon at the beginning of
    each episode.
        epsilon_minimum: Minimum exploration probability. 

    Returns:
        A tuple containing:
            evaluation_network: Trained Q-network.
            target_network: Final target network.
            episode_returns: Total reward collected in each episode.
    """
    # Setting number of threads to 1
    torch.set_num_threads(1)

    # Observation: 4 values, action: 2 values
    if Q_network is None:
        evaluation_network = QNetwork(4, 2)
    else:
        evaluation_network = Q_network

    target_network = QNetwork(4, 2)
    target_network.load_state_dict(evaluation_network.state_dict())
    target_network.eval()

    # Initialising the optimizer and replay buffer
    optimizer = optim.Adam(evaluation_network.parameters(), lr=alpha)
    replay_buffer = ReplayBuffer(buffer_size)

    if pole_lengths is None:
        pole_lengths = np.linspace(0.4, 1.8, 30)

    # Queue is necessary to communicate between the actor and learner, as they are in separate processes
    context = mp.get_context("spawn")
    experience_queue = context.Queue(maxsize=8)
    weights_queue = context.Queue()

    # Start a separate actor process to play CartPole
    actor = context.Process(
        target=actor_worker,
        args=(experience_queue, weights_queue, snapshot(evaluation_network),
              episodes, pole_lengths, epsilon, epsilon_decay,
              epsilon_minimum)
    )

    # Training statistics
    episode_returns = []
    training_records = []
    total_env_steps = 0
    gradient_updates = 0

    actor.start()

    try:
        # CartPole is played by the actor and the 
        # transitions are sent from the actor to the learner
        # through the experience_queue
        while True:
            try:
                message = experience_queue.get(timeout=30)
            except Empty:
                if not actor.is_alive():
                    raise RuntimeError("The actor process stopped unexpectedly")
                continue

            if message[0] == "done":
                break

            if message[0] == "error":
                raise RuntimeError("Actor error: " + message[1])

            if message[0] == "transitions":
                # The actor can send 32 transitions per message
                for state, action, reward, next_state, terminated in message[1]:
                    replay_buffer.add_transition(
                        state, action, reward, next_state, terminated
                    )
                    total_env_steps += 1

                    # Same DQN update as vanilla, once the buffer is large enough
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample_batch(batch_size)
                        states, actions, rewards, next_states, dones = zip(*batch)

                        states_t = torch.FloatTensor(np.stack(states))
                        actions_t = torch.LongTensor(actions).unsqueeze(1)
                        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                        next_states_t = torch.FloatTensor(np.stack(next_states))
                        dones_t = torch.FloatTensor(dones).unsqueeze(1)

                        # Predict Q-values and select the value of each action taken
                        current_q = evaluation_network(states_t).gather(1, actions_t)

                        # Calculate TD targets without updating target-network weights
                        with torch.no_grad():
                            next_max = target_network(next_states_t).max(1)[0].unsqueeze(1)
                            td_target = rewards_t + gamma * next_max * (1 - dones_t)

                        # Calculation of the loss
                        loss = nn.MSELoss()(current_q, td_target)

                        # Backpropagation, optimization + Gradient clipping
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(evaluation_network.parameters(), 1.0)
                        optimizer.step()
                        gradient_updates += 1

            elif message[0] == "episode":
                episode, episode_reward, epsilon_copy = message[1:]
                episode_returns.append(episode_reward)

                # Updating the target network after x amounts of episodes, by copying 
                # the actor's weights on target network
                if episode % update_target_network == 0:
                    target_network.load_state_dict(evaluation_network.state_dict())
                    weights_queue.put(snapshot(evaluation_network))

                training_records.append((
                    episode, episode_reward, total_env_steps,
                    gradient_updates, epsilon_copy
                ))

                # Printing progress
                if episode % 50 == 0:
                    avg_reward = np.mean(episode_returns[-100:])
                    print(f"Episode {episode}/{episodes} | "
                          f"Avg Reward: {avg_reward:.1f} | "
                          f"Epsilon: {epsilon_copy:.3f} | "
                          f"Steps: {total_env_steps} | "
                          f"Updates: {gradient_updates}", flush=True)

    finally:
        # Making sure the separate actor process stops correctly before continuing
        if actor.is_alive():
            actor.terminate()
            actor.join()
        experience_queue.close()
        weights_queue.close()

    # Saving the model in weights/
    torch.save(evaluation_network.state_dict(), "weights/semih_async_dqn.pth")

    # Storing the data needed for evaluation
    with open("results/data/semih_async_training.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "total_env_steps", "gradient_updates", "epsilon"])
        writer.writerows(training_records)

    print("Saved: weights/semih_async_dqn.pth")
    print("Saved: results/data/semih_async_training.csv")

    return (evaluation_network, target_network, episode_returns)


if __name__ == "__main__":

    trained_network, target_network, rewards = main(
        epsilon=1.0,
        gamma=0.99,
        alpha=0.001,
        Q_network=None,
        episodes=500,
        pole_lengths=None,
        batch_size=64,
        buffer_size=50000,
        update_target_network=10,
        epsilon_decay=0.995,
        epsilon_minimum=0.1
    )
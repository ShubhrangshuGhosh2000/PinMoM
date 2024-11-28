import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QNetwork(nn.Module):
    """
    Defines the Q-Network for DQN. It takes the state as input and outputs Q-values for each action.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Q-Network.
        
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)


    def forward(self, x):
        """
        Forward pass through the network.
        
        :param x: Input state.
        :return: Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
# End of class QNetwork


class DQNAgent:
    """
    Defines the DQN Agent which interacts with the environment and learns the optimal policy.
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, epsilon_reset=1.0
                 , lr=0.001, batch_size=64, memory_size=10000, target_update_freq=10):
        """
        Initialize the DQN Agent.
        
        :param state_dim: Dimension of the state space.
        :param action_dim: Dimension of the action space.
        :param gamma: Discount factor for future rewards.
        :param epsilon: Initial exploration rate.
        :param epsilon_min: Minimum exploration rate.
        :param epsilon_decay: Decay rate for exploration probability.
        :param epsilon_reset: Reset value for exploration rate.
        :param lr: Learning rate for the optimizer.
        :param batch_size: Size of the batch for experience replay.
        :param memory_size: Maximum size of the replay memory.
        :param target_update_freq: Frequency to update the target network.
        """
        
        # #######################################  Theory from the book - Start (TO BE REMOVED LATER) ##############################
        #
        # Q-Value Intuition
        # ========================
        # The Q-value Q(s,a) represents the expected cumulative reward of taking action a in state s and then following the optimal policy thereafter. 
        # This is achieved through an iterative update process where the Q-values are refined over time based on the rewards received and the estimated future rewards.
        
        # Immediate Reward: The reward r received immediately after taking action a in state s.
        # Future Rewards: The discounted sum of rewards the agent expects to receive in the future after transitioning to the next state s′.
        # ------------------
        # Bellman Equation
        # -------------------
        # Q(s,a) ← Q(s,a) + α[r + γ max_a′ ​Q(s′,a′) − Q(s,a)]
        # Breaking down this equation:
        # Current Q-value Q(s,a): The current estimate of the Q-value for state s and action a.
        # Learning Rate α: Determines how much the new information overrides the old information. A higher α means new information is given more weight.
        # Immediate Reward r: The reward received after taking action a in state s.
        # Discount Factor γ: Determines the importance of future rewards. A value of γ close to 1 makes future rewards more significant, while a value close to 0 makes the agent short-sighted.
        # Future Q-value max_a′ Q(s′,a′): The maximum Q-value for the next state s′, representing the best possible future reward.
        #
        # =========================
        # Deep Q-Learning Overview
        # =========================
        # Deep Q-Learning replaces the Q-table with a neural network, known as a Q-network, which approximates the Q-value function. 
        # The Q-network takes the state as input and outputs Q-values for all possible actions. 
        # This allows the agent to handle large or continuous state spaces efficiently.
        #
        # Key Components
        # ---------------
        # Q-Network: A neural network that approximates the Q-value function Q(s,a;θ), where θ are the network parameters.
        # Experience Replay: A memory buffer that stores the agent's experiences (s,a,r,s′,done). Experiences are sampled randomly from this buffer to train the Q-network.
        # Target Network: A separate network, periodically updated with the Q-network’s weights, to provide stable target values during training.
        # 
        # ----------------
        # DQN Algorithm
        # ----------------
        # 1. Initialize:
        #     Q-network with random weights θ.
        #     Target network with weights θ' ← θ.
        #     Experience replay buffer.
        #
        # 2. Interaction with the Environment:
        #     For each episode:
        #         Initialize the starting state s.
        #         For each step within the episode:
        #             Choose action a using an exploration strategy (e.g., ε-greedy) based on Q-values from the Q-network.
        #             Execute action a, observe reward r and next state s′.
        #             Store experience (s,a,r,s′,done) in the replay buffer.
        #             Sample a random mini-batch of experiences from the replay buffer.
        #             Compute the target Q-value:
        #                 y = r (if done)
        #                     r + γ max_a′ ​Q(s′,a′) (Otherwise)
        #                 
        #              Perform a gradient descent step on the loss:
        #                 L(θ) = (y − Q(s,a;θ))^2
        #              Periodically update the target network:
        #                 θ' ← θ
        #
        # #######################################  Theory from the book - End (TO BE REMOVED LATER) ##############################

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_reset = epsilon_reset
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        self.steps = 0


    def act(self, state):
        """
        Select an action based on the current state.
        
        :param state: Current state.
        :return: Selected action.
        """
        # Epsilon-greedy (ε-greedy) strategy
        if np.random.rand() <= self.epsilon:
            # Exploration
            return random.randrange(self.action_dim)
        # Otherwise exploitation
        state = torch.FloatTensor(state).unsqueeze(0)
        self.q_network.eval()
        q_values = None
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()


    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        :param state: Previous state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: State after taking the action.
        :param done: Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))


    def replay(self):
        """
        Perform experience replay to train the Q-network.
        
        :return: Loss of the replay step, or None if there are not enough samples in memory.
        """
        # Ensures that if there are not enough samples in the replay memory, the method returns None.
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()


    def reset_epsilon(self):
        """
        Reset the epsilon value to its initial reset value.
        """
        self.epsilon = self.epsilon_reset


    def load(self, name):
        """
        Load the Q-network from a file.
        
        :param name: File name to load the Q-network.
        """
        self.q_network.load_state_dict(torch.load(name))


    def save(self, name):
        """
        Save the Q-network to a file.
        
        :param name: File name to save the Q-network.
        """
        torch.save(self.q_network.state_dict(), name)
# End of class DQNAgent


def compute_stability(sequence):
    """
    Compute the stability of a given protein sequence.
    
    :param sequence: Protein sequence.
    :return: Stability score.
    """
    # Placeholder for actual stability computation
    # In practice, this would involve a sophisticated model or simulation
    # TODO 
    return np.random.rand()  # Random value for illustration


def perform_action(state, action, decreasing):
    """
    Perform an action on the current state to get the next state, reward, and done flag.
    The action_index is used to deterministically perform the mutations.

    :param state: Current state.
    :param action_index: Index representing the action to be taken.
    :param decreasing: Boolean flag representing whether the agent is currently on 'dcreasing' or 'increasing' phase of its journey.
    :return: Next state, best_stability_score, and done flag.
    """
    best_next_state = state.copy()
    best_stability_score = 1.0 if decreasing else 0.0

    for _ in range(5):  # Perform 5 mutation instances
        next_state = state.copy()
        # Deterministic Mutations: The action_index is used as a seed for np.random, ensuring 
        # that the mutations are reproducible and deterministic for the same action_index. 
        # This behavior aligns with the principles of DQL, where actions need to have consistent and 
        # reproducible effects on the state.
        np.random.seed(action)  # Use action_index as the seed for reproducibility
        mutation_positions = np.random.choice(len(state), size=3, replace=False)
        new_amino_acids = np.random.randint(0, 20, size=3)  # Perform 3 mutations in each instance

        for pos, new_aa in zip(mutation_positions, new_amino_acids):
            next_state[pos] = new_aa

        stability_score = compute_stability(next_state)

        if(decreasing and (stability_score < best_stability_score)):
            # For the 'decreasing' phase of the journey, minimum of all the stability_score(s) is considered
            best_stability_score = stability_score
            best_next_state = next_state
        elif((not decreasing) and (stability_score > best_stability_score)):
            # For the 'increasing' phase of the journey, maximum of all the stability_score(s) is considered
            best_stability_score = stability_score
            best_next_state = next_state
    # End of for loop: for _ in range(5):

    # ### reward = compute_reward(best_stability_score, decreasing)
    done = False  # In this case, the episode continues indefinitely or based on a condition
    return best_next_state, best_stability_score, done


def compute_reward(current_stability, decreasing):
    """Compute the reward based on the current_stability and decreasing flag.
    """
    if decreasing:
        # Higher reward for lower stability when decreasing
        return -current_stability
    else:
        # Higher reward for higher stability when increasing
        return current_stability


def train_dqn_agent(agent, initial_sequence, num_episodes, max_steps, convergence_threshold, patience):
    """
    Train the DQN agent on the environment.
    
    :param agent: DQN agent.
    :param initial_sequence: Initial protein sequence.
    :param num_episodes: Number of episodes for training.
    :param max_steps: Maximum steps per episode.
    :param convergence_threshold: Convergence threshold for training.
    :param patience: Number of episodes to check for convergence.
    """
    loss_history = deque(maxlen=patience)

    for episode in range(num_episodes):
        # #### Starting each episode either from the same initial state or the last state of the previous episode 
        state = initial_sequence.copy()  # Reset to fixed initial state
        # state = initial_sequence.copy() if episode == 0 else state  # Use the last state from the previous episode
        
        # Initial stability score needs to be considered
        stability = compute_stability(state)

        decreasing = True  # Start with decreasing stability
        episode_loss = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, current_stability, done = perform_action(state, action, decreasing)
            # Compare current_stability with stability and based on decreasing flag decide whether to skip this step
            if((decreasing and current_stability >= stability) or (not decreasing and current_stability <= stability)):
                # Skip this step
                continue
            else:
                stability = current_stability

            reward = compute_reward(current_stability, decreasing)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # Switch to increasing stability if the target is reached
            if decreasing and current_stability <= 0.5:
                decreasing = False
                agent.reset_epsilon()  # Reset epsilon
            elif not decreasing and current_stability > 0.99:
                decreasing = True
                agent.reset_epsilon()  # Reset epsilon
            # End of if-else block

            # Replay and get loss if memory has enough samples
            loss = agent.replay()
            if loss is not None:
                episode_loss += loss
            if done:
                break
        # End of for loop: for step in range(max_steps):

        # Only append average loss if there was a replay i.e. loss is not None
        if loss is not None:
            avg_loss = episode_loss / max_steps
            loss_history.append(avg_loss)
        
        # Check convergence based on loss history.
        # Convergence Check: Checks if the difference between the maximum and minimum loss in 
        # the most recent 'patience' number of episodes is below the 'convergence_threshold'.
        if len(loss_history) == patience:  # Note that loss_history is a dequeue with maxlen=patience
            if max(loss_history) - min(loss_history) < convergence_threshold:
                print(f"Training converged at episode {episode}")
                break
            # End of if block
        # End of if block: if len(loss_history) == patience:

        if(episode % 5 == 0):
            print(f"Episode {episode}/{num_episodes} completed")
    # End of for loop: for episode in range(num_episodes):

    # Save the agent
    agent.save("dqn_model.pth")
    print("Training completed and model saved.")


def inference_dqn(agent, initial_sequence, max_steps, stability_threshold=0.5):
    """
    Perform inference using the trained DQN agent to find stable sequences.
    
    :param agent: DQN agent.
    :param initial_sequence: Initial protein sequence.
    :param max_steps: Maximum steps for inference.
    :param stability_threshold: Threshold for considering a sequence as stable.
    :return: List of stable sequences found during inference.
    """
    stable_sequences = []
    state = initial_sequence.copy()
    
    decreasing = True  # Start with decreasing stability
    for step in range(max_steps):
        action = agent.act(state)
        next_state, current_stability, done = perform_action(state, action, decreasing)
        if current_stability > stability_threshold:
            stable_sequences.append(next_state)
        state = next_state
        # Switch to increasing stability if the target is reached
        if decreasing and current_stability <= 0.5:
            decreasing = False
        elif not decreasing and current_stability > 0.99:
            decreasing = True
        # End of if-else block
        if done:
            break
    # End of for loop: for step in range(max_steps):

    print("Stable sequences found during inference:", stable_sequences)
    return stable_sequences

def main():
    """
    Main function to initialize the agent, train it, and perform inference.
    """
    # Parameters
    state_dim = 100  # Example state dimension (length of protein sequence)
    action_dim = state_dim * 20  # Example action dimension (20 possible amino acids per position)
    num_episodes = 1000
    max_steps = 100
    convergence_threshold = 1e-3
    patience = 10
    stability_threshold = -0.5

    # Initialize the agent
    agent = DQNAgent(state_dim, action_dim)

    # Initial highly stable sequence (encoded as integers for each amino acid)
    initial_sequence = np.random.randint(0, 20, size=state_dim)

    # Train the agent
    train_dqn_agent(agent, initial_sequence, num_episodes, max_steps, convergence_threshold, patience)

    # Load the model for inference
    agent.load("dqn_model.pth")

    # Perform inference
    stable_sequences = inference_dqn(agent, initial_sequence, max_steps, stability_threshold)

if __name__ == "__main__":
    main()

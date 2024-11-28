import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, batch_size=64, memory_size=10000, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.target_update_freq = target_update_freq
        self.steps = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
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
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def load(self, name):
        self.q_network.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.q_network.state_dict(), name)

# Example stability function
def compute_stability(sequence):
    # Placeholder for actual stability computation
    # In practice, this would involve a sophisticated model or simulation
    return np.random.rand()  # Random value for illustration

# Environment interaction function
def perform_action(state, action):
    best_next_state = state.copy()
    best_stability_score = float('inf')
    
    for _ in range(3):  # Perform 3 mutation instances
        next_state = state.copy()
        n_mutations = np.random.randint(1, 11)  # Randomly select n between 1 and 10
        
        for _ in range(n_mutations):
            position = np.random.randint(0, len(state))  # Randomly select a position
            new_amino_acid = np.random.randint(0, 20)  # Randomly select a new amino acid
            next_state[position] = new_amino_acid
        
        stability_score = compute_stability(next_state)
        
        if stability_score < best_stability_score:
            best_stability_score = stability_score
            best_next_state = next_state

    reward = -best_stability_score  # Higher reward for lower stability score
    done = False  # In this case, the episode continues indefinitely or based on a condition
    return best_next_state, reward, done

def compute_reward(current_stability, target_stability, decreasing):
    if decreasing:
        # Higher reward for lower stability when decreasing
        return -current_stability
    else:
        # Higher reward for higher stability when increasing
        return current_stability

def train_dqn(agent, initial_sequence, num_episodes, max_steps, convergence_threshold, patience):
    loss_history = deque(maxlen=patience)

    for episode in range(num_episodes):
        state = initial_sequence.copy()  # Reset to fixed initial state
        decreasing = True  # Start with decreasing stability
        total_loss = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = perform_action(state, action)
            current_stability = compute_stability(state)
            reward = compute_reward(current_stability, 0.5 if decreasing else 1.0, decreasing)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            total_loss += loss
            state = next_state
            if done:
                break

            # Switch to increasing stability if the target is reached
            if decreasing and current_stability <= 0.5:
                decreasing = False
            elif not decreasing and current_stability > 0.99:
                decreasing = True
            
        avg_loss = total_loss / max_steps
        loss_history.append(avg_loss)
        
        # Check for convergence
        if len(loss_history) == patience:
            if max(loss_history) - min(loss_history) < convergence_threshold:
                print(f"Training converged at episode {episode}")
                break

    agent.save("dqn_model.pth")
    print("Training completed and model saved.")

def inference_dqn(agent, initial_sequence, max_steps, stability_threshold=0.5):
    stable_sequences = []
    state = initial_sequence.copy()
    decreasing = True

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = perform_action(state, action)
        # current_stability =

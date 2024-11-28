import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

# Certainly! Below is the full code incorporating the heuristic 
# where the agent will move to the next state only if the stability criterion is met (depending on the decreasing flag) and occasionally
#  allow transitions that do not meet this criterion to ensure exploration. The code is structured with appropriate comments,
#  encapsulation, and includes training and inference methods.

# Key Changes and Enhancements:

# Heuristic Stability Check: In both train_agent and inference, the code checks if the stability is increasing or decreasing according to the 
# decreasing flag. It only allows moving to the next state if the criterion is met, with a small probability to allow exceptions.
# Perturbation: Occasionally, transitions that do not meet the stability criterion are allowed to ensure exploration.
# Stability Check and Transition: During inference, stable sequences are recorded if their stability is above 0.5, regardless of the decreasing flag.
# Training and Inference Functions: These functions encapsulate the main logic and can be called from main() to run the training and inference processes.


# Define a simple neural network for the Q-learning agent
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

# Define the Q-learning agent
class Agent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, lr, tau, update_every):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.update_every = update_every

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))
        self.qnetwork_target.load_state_dict(torch.load(path))

# Function to calculate stability of a protein sequence
def calculate_stability(sequence):
    return random.uniform(0, 1)

# Function to perform mutation on a protein sequence
def perform_action(sequence, action):
    sequence = list(sequence)
    action = random.sample(range(len(sequence)), random.randint(1, 10))
    for idx in action:
        sequence[idx] = random.choice('ACDEFGHIKLMNPQRSTVWY')
    return ''.join(sequence)

# Training the agent
def train_agent(agent, num_episodes, max_timesteps, epsilon_start, epsilon_end, epsilon_decay):
    initial_sequence = "M" * 100
    stable_sequences = []

    for i_episode in range(1, num_episodes + 1):
        state = initial_sequence
        stability = calculate_stability(state)
        decreasing = True
        eps = max(epsilon_end, epsilon_start * (epsilon_decay ** i_episode))
        
        for t in range(max_timesteps):
            action = agent.act(np.array([ord(c) for c in state]), eps)
            next_state = perform_action(state, action)
            next_stability = calculate_stability(next_state)

            if decreasing and next_stability > stability and random.random() > 0.1:
                next_state, next_stability = state, stability
            elif not decreasing and next_stability < stability and random.random() > 0.1:
                next_state, next_stability = state, stability

            reward = next_stability if not decreasing else -next_stability
            done = next_stability <= 0.5 if decreasing else next_stability >= 0.99

            agent.step(np.array([ord(c) for c in state]), action, reward, np.array([ord(c) for c in next_state]), done)
            state, stability = next_state, next_stability

            if decreasing and stability > 0.99:
                decreasing = False

            if done:
                break
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}/{num_episodes} completed")

    agent.save('dqn_model.pth')

# Running inference
def inference(agent, initial_sequence, num_steps):
    agent.load('dqn_model.pth')
    state = initial_sequence
    stability = calculate_stability(state)
    stable_sequences = []
    decreasing = stability > 0.99

    for t in range(num_steps):
        action = agent.act(np.array([ord(c) for c in state]))
        next_state = perform_action(state, action)
        next_stability = calculate_stability(next_state)

        if decreasing and next_stability > stability and random.random() > 0.1:
            next_state, next_stability = state, stability
        elif not decreasing and next_stability < stability and random.random() > 0.1:
            next_state, next_stability = state, stability

        if next_stability > 0.5:
            stable_sequences.append(next_state)
        
        state, stability = next_state, next_stability

        if decreasing and stability > 0.99:
            decreasing = False

    return stable_sequences

# Main method
def main():
    state_size = 100
    action_size = 20  # Number of possible mutations
    buffer_size = int(1e5)
    batch_size = 64
    gamma = 0.99
    lr = 0.001
    tau = 1e-3
    update_every = 4

    agent = Agent(state_size, action_size, buffer_size, batch_size, gamma, lr, tau, update_every)

    num_episodes = 2000
    max_timesteps = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    train_agent(agent, num_episodes, max_timesteps, epsilon_start, epsilon_end, epsilon_decay)
    
    initial_sequence = "M" * 100
    num_steps = 1000
    stable_sequences = inference(agent, initial_sequence, num_steps)
    print(f"Stable sequences found: {stable_sequences}")

if __name__ == "__main__":
    main()

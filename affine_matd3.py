"""
Affine-Aware MATD3 for Distributed Formation Control
====================================================

Key Modifications:
1. Actors output affine parameter votes (not direct velocities)
2. Critics see global affine state
3. Coordination reward for vote alignment
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AffineReplayBuffer:
    """Replay buffer that stores global affine state"""
    
    def __init__(self, buffer_size, num_agents, obs_dims, action_dims, device):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Storage
        self.observations = [
            np.zeros((buffer_size, obs_dims[i]), dtype=np.float32)
            for i in range(num_agents)
        ]
        self.next_observations = [
            np.zeros((buffer_size, obs_dims[i]), dtype=np.float32)
            for i in range(num_agents)
        ]
        self.actions = [
            np.zeros((buffer_size, action_dims[i]), dtype=np.float32)
            for i in range(num_agents)
        ]
        self.rewards = [
            np.zeros((buffer_size, 1), dtype=np.float32)
            for i in range(num_agents)
        ]
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        
        # NEW: Store global affine state
        self.affine_states = np.zeros((buffer_size, 4), dtype=np.float32)
        self.next_affine_states = np.zeros((buffer_size, 4), dtype=np.float32)
    
    def add(self, obs, next_obs, actions, rewards, done, affine_state, next_affine_state):
        for i in range(self.num_agents):
            self.observations[i][self.ptr] = obs[i]
            self.next_observations[i][self.ptr] = next_obs[i]
            self.actions[i][self.ptr] = actions[i]
            self.rewards[i][self.ptr] = rewards[i]
        
        self.dones[self.ptr] = done
        self.affine_states[self.ptr] = affine_state
        self.next_affine_states[self.ptr] = next_affine_state
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch_obs = [
            torch.FloatTensor(self.observations[i][idxs]).to(self.device)
            for i in range(self.num_agents)
        ]
        batch_next_obs = [
            torch.FloatTensor(self.next_observations[i][idxs]).to(self.device)
            for i in range(self.num_agents)
        ]
        batch_actions = [
            torch.FloatTensor(self.actions[i][idxs]).to(self.device)
            for i in range(self.num_agents)
        ]
        batch_rewards = [
            torch.FloatTensor(self.rewards[i][idxs]).to(self.device)
            for i in range(self.num_agents)
        ]
        batch_dones = torch.FloatTensor(self.dones[idxs]).to(self.device)
        batch_affine = torch.FloatTensor(self.affine_states[idxs]).to(self.device)
        batch_next_affine = torch.FloatTensor(self.next_affine_states[idxs]).to(self.device)
        
        return (batch_obs, batch_next_obs, batch_actions, batch_rewards, 
                batch_dones, batch_affine, batch_next_affine)


class AffineVotingActor(nn.Module):
    """
    Actor that outputs affine parameter votes
    [delta_rotation, delta_scale, delta_vx, delta_vy]
    """
    
    def __init__(self, obs_dim, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        # Initialize with small weights
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        vote = torch.tanh(self.fc3(x))  # Bounded [-1, 1]
        return vote


class AffineAwareCritic(nn.Module):
    """
    Critic that sees global affine state
    Input: all_obs + all_actions + global_affine
    """
    
    def __init__(self, total_obs_dim, total_action_dim, affine_dim=4):
        super().__init__()
        input_dim = total_obs_dim + total_action_dim + affine_dim
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
    
    def forward(self, all_obs, all_actions, global_affine):
        x = torch.cat([all_obs, all_actions, global_affine], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OrnsteinUhlenbeckNoise:
    """OU process for exploration"""
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma_start=0.3, 
                 sigma_min=0.03, decay_steps=60000):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.decay_steps = decay_steps
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def get_sigma(self, step):
        if step < self.decay_steps:
            return self.sigma_start * np.exp(-step / (self.decay_steps / 4))
        return self.sigma_min
    
    def set_sigma(self, new_sigma):
        self.sigma_start = new_sigma
        self.reset()
    
    def __call__(self, step):
        sigma = self.get_sigma(step)
        self.state += self.theta * (self.mu - self.state) + \
                     sigma * np.random.randn(self.action_dim)
        return self.state.copy()


class AffineMatD3:
    """Multi-Agent TD3 for Affine Formation Control"""
    
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_frequency=2,
                 buffer_size=1000000, batch_size=256, learning_starts=25000,
                 device="cuda", seed=42):
        
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get environment info
        env.reset(seed=seed)
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        self.obs_dims = [env.observation_spaces[agent].shape[0] for agent in self.agents]
        self.action_dims = [4 for _ in self.agents]  # All vote on 4 affine params
        
        total_obs_dim = sum(self.obs_dims)
        total_action_dim = sum(self.action_dims)
        
        # Create actors (vote on affine parameters)
        self.actors = [
            AffineVotingActor(self.obs_dims[i], 4).to(self.device)
            for i in range(self.num_agents)
        ]
        self.target_actors = [
            AffineVotingActor(self.obs_dims[i], 4).to(self.device)
            for i in range(self.num_agents)
        ]
        
        # Initialize targets
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
        
        # OU noise for exploration
        self.ou_noises = [
            OrnsteinUhlenbeckNoise(action_dim=4, sigma_start=0.3, sigma_min=0.03)
            for _ in range(self.num_agents)
        ]
        
        # Create critics (affine-aware)
        self.critics_1 = [
            AffineAwareCritic(total_obs_dim, total_action_dim, affine_dim=4).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.critics_2 = [
            AffineAwareCritic(total_obs_dim, total_action_dim, affine_dim=4).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics_1 = [
            AffineAwareCritic(total_obs_dim, total_action_dim, affine_dim=4).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics_2 = [
            AffineAwareCritic(total_obs_dim, total_action_dim, affine_dim=4).to(self.device)
            for _ in range(self.num_agents)
        ]
        
        # Initialize targets
        for i in range(self.num_agents):
            self.target_critics_1[i].load_state_dict(self.critics_1[i].state_dict())
            self.target_critics_2[i].load_state_dict(self.critics_2[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(self.actors[i].parameters(), lr=learning_rate)
            for i in range(self.num_agents)
        ]
        self.critic_optimizers = [
            optim.Adam(
                list(self.critics_1[i].parameters()) + list(self.critics_2[i].parameters()),
                lr=learning_rate
            )
            for i in range(self.num_agents)
        ]
        
        # Replay buffer
        self.replay_buffer = AffineReplayBuffer(
            buffer_size, self.num_agents, self.obs_dims, self.action_dims, self.device
        )
        
        self.global_step = 0
    
    def get_actions(self, observations, explore=True):
        """Get affine parameter votes from all agents"""
        actions = []
        for i in range(self.num_agents):
            obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vote = self.actors[i](obs)
                
                if explore and self.global_step >= self.learning_starts:
                    noise = self.ou_noises[i](self.global_step)
                    noise = torch.FloatTensor(noise).to(self.device)
                    vote = torch.clamp(vote + noise, -1, 1)
            
            actions.append(vote.cpu().numpy()[0])
        
        return actions
    
    def update(self):
        """Update all agents"""
        if self.replay_buffer.size < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, actions, rewards, dones, affine, next_affine = batch
        
        # Concatenate
        obs_cat = torch.cat(obs, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)
        actions_cat = torch.cat(actions, dim=1)
        
        losses = {}
        
        for agent_idx in range(self.num_agents):
            # === Update Critic ===
            with torch.no_grad():
                # Get next votes from target actors
                next_votes = []
                for i in range(self.num_agents):
                    next_vote = self.target_actors[i](next_obs[i])
                    
                    # Add clipped noise
                    noise = (torch.randn_like(next_vote) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )
                    next_vote = torch.clamp(next_vote + noise, -1, 1)
                    next_votes.append(next_vote)
                
                next_votes_cat = torch.cat(next_votes, dim=1)
                
                # Target Q values
                target_q1 = self.target_critics_1[agent_idx](
                    next_obs_cat, next_votes_cat, next_affine
                )
                target_q2 = self.target_critics_2[agent_idx](
                    next_obs_cat, next_votes_cat, next_affine
                )
                target_q = torch.min(target_q1, target_q2)
                y = rewards[agent_idx] + (1 - dones) * self.gamma * target_q
            
            # Current Q values
            current_q1 = self.critics_1[agent_idx](obs_cat, actions_cat, affine)
            current_q2 = self.critics_2[agent_idx](obs_cat, actions_cat, affine)
            
            # Critic loss
            critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
            
            # Optimize critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critics_1[agent_idx].parameters()) +
                list(self.critics_2[agent_idx].parameters()),
                max_norm=0.5
            )
            self.critic_optimizers[agent_idx].step()
            
            losses[f'agent_{agent_idx}_critic_loss'] = critic_loss.item()
            
            # === Delayed Actor Update ===
            if self.global_step % self.policy_frequency == 0:
                # Get current votes
                current_votes = []
                for i in range(self.num_agents):
                    if i == agent_idx:
                        vote = self.actors[i](obs[i])
                    else:
                        with torch.no_grad():
                            vote = self.actors[i](obs[i])
                    current_votes.append(vote)
                
                votes_cat = torch.cat(current_votes, dim=1)
                
                # Actor loss
                actor_loss = -self.critics_1[agent_idx](obs_cat, votes_cat, affine).mean()
                
                # Optimize actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), max_norm=0.5)
                self.actor_optimizers[agent_idx].step()
                
                losses[f'agent_{agent_idx}_actor_loss'] = actor_loss.item()
        
        # === Soft Update Targets ===
        if self.global_step % self.policy_frequency == 0:
            for i in range(self.num_agents):
                self._soft_update(self.actors[i], self.target_actors[i])
                self._soft_update(self.critics_1[i], self.target_critics_1[i])
                self._soft_update(self.critics_2[i], self.target_critics_2[i])
        
        return losses
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """Save model"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics_1': [critic.state_dict() for critic in self.critics_1],
            'critics_2': [critic.state_dict() for critic in self.critics_2],
        }, os.path.join(path, 'affine_matd3_model.pth'))
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(os.path.join(path, 'affine_matd3_model.pth'))
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.target_actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics_1[i].load_state_dict(checkpoint['critics_1'][i])
            self.critics_2[i].load_state_dict(checkpoint['critics_2'][i])
            self.target_critics_1[i].load_state_dict(checkpoint['critics_1'][i])
            self.target_critics_2[i].load_state_dict(checkpoint['critics_2'][i])
        print(f"Models loaded from {path}")

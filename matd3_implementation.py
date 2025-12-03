"""
MATD3 (Multi-Agent TD3) Implementation
Clean class-based implementation for multi-agent reinforcement learning
Compatible with PettingZoo parallel environments
"""

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent environments"""
    
    def __init__(self, buffer_size, num_agents, obs_dims, action_dims, device):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Store observations and actions per agent
        self.observations = [
            np.zeros((buffer_size, obs_dim), dtype=np.float32) 
            for obs_dim in obs_dims
        ]
        self.next_observations = [
            np.zeros((buffer_size, obs_dim), dtype=np.float32) 
            for obs_dim in obs_dims
        ]
        self.actions = [
            np.zeros((buffer_size, action_dim), dtype=np.float32) 
            for action_dim in action_dims
        ]
        self.rewards = [
            np.zeros((buffer_size, 1), dtype=np.float32) 
            for _ in range(num_agents)
        ]
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, obs, next_obs, actions, rewards, done):
        for i in range(self.num_agents):
            self.observations[i][self.ptr] = obs[i]
            self.next_observations[i][self.ptr] = next_obs[i]
            self.actions[i][self.ptr] = actions[i]
            self.rewards[i][self.ptr] = rewards[i]
        self.dones[self.ptr] = done
        
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
        
        return batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones


class Actor(nn.Module):
    """Actor network for a single agent - takes only local observation"""
    
    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Action rescaling
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((action_high - action_low) / 2.0)
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((action_high + action_low) / 2.0)
        )
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Critic network - takes all observations and all actions (centralized)"""
    
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MATD3:
    """Multi-Agent TD3 implementation"""
    
    def __init__(
        self,
        env,
        learning_rate=1e-3,
        gamma=0.95,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
        policy_frequency=2,
        buffer_size=1000000,
        batch_size=1024,
        learning_starts=25000,
        hidden_dim=256,
        device="cuda",
        seed=1
    ):
        """
        Initialize MATD3 agent
        
        Args:
            env: PettingZoo parallel environment
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            tau: Target network update rate
            policy_noise: Noise added to target policy for smoothing
            noise_clip: Clip range for policy noise
            exploration_noise: Noise added during exploration
            policy_frequency: Frequency of delayed policy updates
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            learning_starts: Number of steps before training starts
            hidden_dim: Hidden dimension for networks
            device: Device to run on ('cuda' or 'cpu')
            seed: Random seed
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_frequency = policy_frequency
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get agent information
        env.reset(seed=seed)
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        self.obs_dims = [env.observation_space(agent).shape[0] for agent in self.agents]
        self.action_dims = [env.action_space(agent).shape[0] for agent in self.agents]
        self.action_lows = [env.action_space(agent).low for agent in self.agents]
        self.action_highs = [env.action_space(agent).high for agent in self.agents]
        
        total_obs_dim = sum(self.obs_dims)
        total_action_dim = sum(self.action_dims)
        
        # Create actors for each agent
        self.actors = [
            Actor(self.obs_dims[i], self.action_dims[i], 
                  self.action_lows[i], self.action_highs[i], hidden_dim).to(self.device)
            for i in range(self.num_agents)
        ]
        self.target_actors = [
            Actor(self.obs_dims[i], self.action_dims[i], 
                  self.action_lows[i], self.action_highs[i], hidden_dim).to(self.device)
            for i in range(self.num_agents)
        ]
        
        # Create critics for each agent (2 Q-networks per agent for TD3)
        self.critics_1 = [
            Critic(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.critics_2 = [
            Critic(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics_1 = [
            Critic(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.target_critics_2 = [
            Critic(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
            for _ in range(self.num_agents)
        ]
        
        # Initialize target networks
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics_1[i].load_state_dict(self.critics_1[i].state_dict())
            self.target_critics_2[i].load_state_dict(self.critics_2[i].state_dict())
        
        # Create optimizers
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
        
        # Initialize replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(
            buffer_size, self.num_agents, self.obs_dims, self.action_dims, self.device
        )
        
        # Training stats
        self.global_step = 0
        self.episode_rewards = [0.0 for _ in range(self.num_agents)]
        self.episode_length = 0
    
    def get_actions(self, observations, explore=True):
        """Get actions from all agents"""
        actions = []
        for i in range(self.num_agents):
            obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.actors[i](obs)
                if explore:
                    noise = torch.normal(
                        0, self.actors[i].action_scale * self.exploration_noise
                    ).to(self.device)
                    action = action + noise
                    action = action.clamp(
                        torch.FloatTensor(self.actors[i].action_bias - self.actors[i].action_scale).to(self.device),
                        torch.FloatTensor(self.actors[i].action_bias + self.actors[i].action_scale).to(self.device)
                    )
            actions.append(action.cpu().numpy()[0])
        return actions
    
    def update(self):
        """Update all agents using a batch from replay buffer"""
        if self.replay_buffer.size < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, actions, rewards, dones = batch
        
        # Concatenate all observations and actions
        obs_cat = torch.cat(obs, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)
        actions_cat = torch.cat(actions, dim=1)
        
        losses = {}
        
        for agent_idx in range(self.num_agents):
            # -------- Update Critic --------
            with torch.no_grad():
                # Get next actions from target actors
                next_actions = []
                for i in range(self.num_agents):
                    next_action = self.target_actors[i](next_obs[i])
                    # Add clipped noise
                    noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    ) * self.target_actors[i].action_scale
                    next_action = (next_action + noise).clamp(
                        self.target_actors[i].action_bias - self.target_actors[i].action_scale,
                        self.target_actors[i].action_bias + self.target_actors[i].action_scale
                    )
                    next_actions.append(next_action)
                
                next_actions_cat = torch.cat(next_actions, dim=1)
                
                # Compute target Q-values
                target_q1 = self.target_critics_1[agent_idx](next_obs_cat, next_actions_cat)
                target_q2 = self.target_critics_2[agent_idx](next_obs_cat, next_actions_cat)
                target_q = torch.min(target_q1, target_q2)
                target_q = rewards[agent_idx] + (1 - dones) * self.gamma * target_q
            
            # Get current Q-values
            current_q1 = self.critics_1[agent_idx](obs_cat, actions_cat)
            current_q2 = self.critics_2[agent_idx](obs_cat, actions_cat)
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Optimize critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()
            
            losses[f'agent_{agent_idx}_critic_loss'] = critic_loss.item()
            losses[f'agent_{agent_idx}_q1_value'] = current_q1.mean().item()
            losses[f'agent_{agent_idx}_q2_value'] = current_q2.mean().item()
            
            # -------- Delayed Policy Update --------
            if self.global_step % self.policy_frequency == 0:
                # Compute actor loss
                actions_pred = []
                for i in range(self.num_agents):
                    if i == agent_idx:
                        actions_pred.append(self.actors[i](obs[i]))
                    else:
                        actions_pred.append(actions[i].detach())
                
                actions_pred_cat = torch.cat(actions_pred, dim=1)
                actor_loss = -self.critics_1[agent_idx](obs_cat, actions_pred_cat).mean()
                
                # Optimize actor
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent_idx].step()
                
                losses[f'agent_{agent_idx}_actor_loss'] = actor_loss.item()
                
                # Update target networks
                self._soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
                self._soft_update(self.critics_1[agent_idx], self.target_critics_1[agent_idx])
                self._soft_update(self.critics_2[agent_idx], self.target_critics_2[agent_idx])
        
        return losses
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def train(self, total_timesteps, log_interval=1000, save_path=None, use_tensorboard=False):
        """
        Train the MATD3 agents
        
        Args:
            total_timesteps: Total number of environment steps
            log_interval: Interval for logging statistics
            save_path: Path to save models (None to disable saving)
            use_tensorboard: Whether to use tensorboard logging
        """
        if use_tensorboard:
            run_name = f"MATD3_{int(time.time())}"
            writer = SummaryWriter(f"runs/{run_name}")
        
        start_time = time.time()
        obs_dict, _ = self.env.reset()
        obs = [obs_dict[agent] for agent in self.agents]
        
        self.episode_rewards = [0.0 for _ in range(self.num_agents)]
        self.episode_length = 0
        
        for step in range(total_timesteps):
            self.global_step = step
            
            # Select actions
            if step < self.learning_starts:
                actions = [self.env.action_space(agent).sample() for agent in self.agents]
            else:
                actions = self.get_actions(obs, explore=True)
            
            # Step environment
            action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
            next_obs_dict, reward_dict, termination_dict, truncation_dict, info_dict = self.env.step(action_dict)
            
            next_obs = [next_obs_dict[agent] for agent in self.agents]
            rewards = [reward_dict[agent] for agent in self.agents]
            done = any(termination_dict.values()) or any(truncation_dict.values())
            
            # Track episode rewards
            for i in range(self.num_agents):
                self.episode_rewards[i] += rewards[i]
            self.episode_length += 1
            
            # Store transition
            self.replay_buffer.add(obs, next_obs, actions, rewards, done)
            
            obs = next_obs
            
            # Reset if done
            if done:
                mean_reward = np.mean(self.episode_rewards)
                if step % log_interval == 0:
                    print(f"Step: {step}, Episode Length: {self.episode_length}, Mean Reward: {mean_reward:.2f}")
                
                if use_tensorboard:
                    writer.add_scalar("charts/mean_episodic_return", mean_reward, step)
                    writer.add_scalar("charts/episodic_length", self.episode_length, step)
                
                obs_dict, _ = self.env.reset()
                obs = [obs_dict[agent] for agent in self.agents]
                self.episode_rewards = [0.0 for _ in range(self.num_agents)]
                self.episode_length = 0
            
            # Training
            if step >= self.learning_starts:
                losses = self.update()
                
                if step % log_interval == 0 and losses is not None:
                    sps = int(step / (time.time() - start_time))
                    print(f"SPS: {sps}")
                    
                    if use_tensorboard:
                        writer.add_scalar("charts/SPS", sps, step)
                        for loss_name, loss_value in losses.items():
                            writer.add_scalar(f"losses/{loss_name}", loss_value, step)
        
        # Save models
        if save_path is not None:
            self.save(save_path)
            print(f"Models saved to {save_path}")
        
        if use_tensorboard:
            writer.close()
        
        self.env.close()
    
    def test(self, num_episodes=10, render=False, verbose=True):
        """
        Test the trained agents
        
        Args:
            num_episodes: Number of episodes to test
            render: Whether to render the environment
            verbose: Whether to print episode statistics
        
        Returns:
            mean_rewards: Mean rewards for each agent across episodes
        """
        all_episode_rewards = [[] for _ in range(self.num_agents)]
        
        for episode in range(num_episodes):
            obs_dict, _ = self.env.reset()
            obs = [obs_dict[agent] for agent in self.agents]
            episode_rewards = [0.0 for _ in range(self.num_agents)]
            done = False
            step_count = 0
            
            while not done:
                # Get actions without exploration
                actions = self.get_actions(obs, explore=False)
                
                # Step environment
                action_dict = {agent: actions[i] for i, agent in enumerate(self.agents)}
                next_obs_dict, reward_dict, termination_dict, truncation_dict, _ = self.env.step(action_dict)
                
                next_obs = [next_obs_dict[agent] for agent in self.agents]
                rewards = [reward_dict[agent] for agent in self.agents]
                done = any(termination_dict.values()) or any(truncation_dict.values())
                
                for i in range(self.num_agents):
                    episode_rewards[i] += rewards[i]
                
                obs = next_obs
                step_count += 1
            
            for i in range(self.num_agents):
                all_episode_rewards[i].append(episode_rewards[i])
            
            if verbose:
                mean_reward = np.mean(episode_rewards)
                print(f"Episode {episode + 1}: Length={step_count}, Mean Reward={mean_reward:.2f}")
        
        mean_rewards = [np.mean(agent_rewards) for agent_rewards in all_episode_rewards]
        
        if verbose:
            print("\nTest Results:")
            for i, mean_reward in enumerate(mean_rewards):
                print(f"Agent {i}: Mean Reward = {mean_reward:.2f}")
            print(f"Overall Mean Reward: {np.mean(mean_rewards):.2f}")
        
        return mean_rewards
    
    def save(self, path):
        """Save model weights"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics_1': [critic.state_dict() for critic in self.critics_1],
            'critics_2': [critic.state_dict() for critic in self.critics_2],
        }, os.path.join(path, 'matd3_model.pth'))
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(os.path.join(path, 'matd3_model.pth'))
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.target_actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics_1[i].load_state_dict(checkpoint['critics_1'][i])
            self.critics_2[i].load_state_dict(checkpoint['critics_2'][i])
            self.target_critics_1[i].load_state_dict(checkpoint['critics_1'][i])
            self.target_critics_2[i].load_state_dict(checkpoint['critics_2'][i])
        print(f"Models loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Create environment
    try:
        from pettingzoo.mpe import simple_spread_v3
        env = simple_spread_v3.parallel_env(render_mode=None, continuous_actions=True)
    except ImportError:
        raise ImportError("Please install PettingZoo: pip install pettingzoo[mpe]")
    
    # Initialize MATD3
    matd3 = MATD3(
        env=env,
        learning_rate=1e-3,
        gamma=0.95,
        tau=0.01,
        batch_size=1024,
        learning_starts=25000,
        device="cuda",
        seed=1
    )
    
    # Train
    print("Training MATD3...")
    matd3.train(
        total_timesteps=1000000,
        log_interval=10000,
        save_path="./models",
        use_tensorboard=True
    )
    
    # Test
    print("\nTesting MATD3...")
    matd3.test(num_episodes=10, verbose=True)

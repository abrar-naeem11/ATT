import numpy as np
import torch


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent environments"""
    
    def __init__(self, buffer_size, num_agents, obs_dims, action_dims, device):
        """
        Initialize replay buffer for multi-agent settings.
        
        Args:
            buffer_size: Maximum number of transitions to store
            num_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            action_dims: List of action dimensions for each agent
            device: Device to put tensors on (cpu/cuda)
        """
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
        # FIX: Rewards should be scalars, not (buffer_size, 1)
        self.rewards = [
            np.zeros(buffer_size, dtype=np.float32) 
            for _ in range(num_agents)
        ]
        # FIX: Dones should also be 1D
        self.dones = np.zeros(buffer_size, dtype=np.float32)
    
    def add(self, obs, next_obs, actions, rewards, done):
        """
        Add a transition to the buffer.
        
        Args:
            obs: List of observations for each agent
            next_obs: List of next observations for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent (scalars)
            done: Boolean or float indicating if episode is done
        """
        for i in range(self.num_agents):
            self.observations[i][self.ptr] = obs[i]
            self.next_observations[i][self.ptr] = next_obs[i]
            self.actions[i][self.ptr] = actions[i]
            self.rewards[i][self.ptr] = float(rewards[i])  # Ensure scalar
        
        self.dones[self.ptr] = float(done)  # Convert bool to float
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (observations, next_observations, actions, rewards, dones)
            Each element is a list (for agents) of tensors or a tensor (for dones)
        """
        # Sample random indices
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Convert to tensors and move to device
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
        # FIX: Add unsqueeze to make rewards (batch_size, 1) for compatibility
        batch_rewards = [
            torch.FloatTensor(self.rewards[i][idxs]).unsqueeze(1).to(self.device)
            for i in range(self.num_agents)
        ]
        # FIX: Add unsqueeze to make dones (batch_size, 1) for compatibility
        batch_dones = torch.FloatTensor(self.dones[idxs]).unsqueeze(1).to(self.device)
        
        return batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones
    
    def __len__(self):
        """Return current size of buffer"""
        return self.size
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples"""
        return self.size >= batch_size


# Optional: Enhanced version with episode boundary tracking
class MultiAgentReplayBufferWithBoundaries(MultiAgentReplayBuffer):
    """
    Enhanced replay buffer that tracks episode boundaries.
    Useful for algorithms that need to avoid bootstrapping across episodes.
    """
    
    def __init__(self, buffer_size, num_agents, obs_dims, action_dims, device):
        super().__init__(buffer_size, num_agents, obs_dims, action_dims, device)
        # Track which transitions are episode starts
        self.episode_starts = np.zeros(buffer_size, dtype=np.bool_)
        self.current_episode_start = 0
    
    def add(self, obs, next_obs, actions, rewards, done):
        """Add transition and track episode boundaries"""
        # Mark episode start
        if self.ptr == 0 or self.dones[(self.ptr - 1) % self.buffer_size]:
            self.episode_starts[self.ptr] = True
            self.current_episode_start = self.ptr
        else:
            self.episode_starts[self.ptr] = False
        
        super().add(obs, next_obs, actions, rewards, done)
    
    def sample(self, batch_size, sequential=False):
        """
        Sample transitions, optionally ensuring no cross-episode bootstrapping.
        
        Args:
            batch_size: Number of transitions to sample
            sequential: If True, sample sequences that don't cross episode boundaries
        """
        if not sequential:
            return super().sample(batch_size)
        
        # Sample indices that aren't followed by episode boundaries
        valid_idxs = np.where(~self.episode_starts[:self.size])[0]
        if len(valid_idxs) < batch_size:
            # Fallback to regular sampling if not enough valid indices
            return super().sample(batch_size)
        
        idxs = np.random.choice(valid_idxs, size=batch_size, replace=False)
        
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
            torch.FloatTensor(self.rewards[i][idxs]).unsqueeze(1).to(self.device)
            for i in range(self.num_agents)
        ]
        batch_dones = torch.FloatTensor(self.dones[idxs]).unsqueeze(1).to(self.device)
        
        return batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones


# Test the buffer
if __name__ == "__main__":
    # Test basic functionality
    device = torch.device("cpu")
    buffer = MultiAgentReplayBuffer(
        buffer_size=1000,
        num_agents=3,
        obs_dims=[4, 6, 8],
        action_dims=[2, 2, 2],
        device=device
    )
    
    # Add some transitions
    for _ in range(100):
        obs = [np.random.randn(dim) for dim in [4, 6, 8]]
        next_obs = [np.random.randn(dim) for dim in [4, 6, 8]]
        actions = [np.random.randn(2) for _ in range(3)]
        rewards = [np.random.randn() for _ in range(3)]  # Scalars
        done = False
        
        buffer.add(obs, next_obs, actions, rewards, done)
    
    # Sample a batch
    batch = buffer.sample(32)
    batch_obs, batch_next_obs, batch_actions, batch_rewards, batch_dones = batch
    
    print("Buffer test passed!")
    print(f"Buffer size: {len(buffer)}")
    print(f"Sample batch shapes:")
    print(f"  Observations: {[obs.shape for obs in batch_obs]}")
    print(f"  Actions: {[act.shape for act in batch_actions]}")
    print(f"  Rewards: {[rew.shape for rew in batch_rewards]}")
    print(f"  Dones: {batch_dones.shape}")
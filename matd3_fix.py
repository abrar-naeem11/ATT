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
                # Add clipped noise for target policy smoothing
                noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                ) * self.target_actors[i].action_scale
                next_action = (next_action + noise).clamp(
                    self.target_actors[i].action_bias - self.target_actors[i].action_scale,
                    self.target_actors[i].action_bias + self.target_actors[i].action_scale
                )
                next_actions.append(next_action)
            
            next_actions_cat = torch.cat(next_actions, dim=1)
            
            # Compute target Q-values (clipped double Q-learning)
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
        # Optional: add gradient clipping
        # torch.nn.utils.clip_grad_norm_(
        #     list(self.critics_1[agent_idx].parameters()) + 
        #     list(self.critics_2[agent_idx].parameters()), 
        #     max_norm=1.0
        # )
        self.critic_optimizers[agent_idx].step()
        
        losses[f'agent_{agent_idx}_critic_loss'] = critic_loss.item()
        losses[f'agent_{agent_idx}_q1_value'] = current_q1.mean().item()
        losses[f'agent_{agent_idx}_q2_value'] = current_q2.mean().item()
        
        # -------- Delayed Policy Update --------
        if self.global_step % self.policy_frequency == 0:
            # Compute actor loss - only update current agent's policy
            # More efficient: compute other agents' actions without gradients
            with torch.no_grad():
                actions_pred = [
                    self.actors[i](obs[i]) if i != agent_idx else None
                    for i in range(self.num_agents)
                ]
            
            # Compute current agent's action with gradients
            actions_pred[agent_idx] = self.actors[agent_idx](obs[agent_idx])
            
            actions_pred_cat = torch.cat(actions_pred, dim=1)
            actor_loss = -self.critics_1[agent_idx](obs_cat, actions_pred_cat).mean()
            
            # Optimize actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            # Optional: add gradient clipping
            # torch.nn.utils.clip_grad_norm_(
            #     self.actors[agent_idx].parameters(), 
            #     max_norm=1.0
            # )
            self.actor_optimizers[agent_idx].step()
            
            losses[f'agent_{agent_idx}_actor_loss'] = actor_loss.item()
        
        # CRITICAL FIX: Update target networks every step, not just when policy updates
        # This is outside the policy_frequency check
        self._soft_update(self.actors[agent_idx], self.target_actors[agent_idx])
        self._soft_update(self.critics_1[agent_idx], self.target_critics_1[agent_idx])
        self._soft_update(self.critics_2[agent_idx], self.target_critics_2[agent_idx])
    
    return losses
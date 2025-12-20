"""
Training Script for MARL Affine Formation Control
=================================================

For Q1 Journal Submission
"""

import os
import time
import csv
import numpy as np
import torch
from collections import deque


def train_affine_matd3(
    env,
    total_timesteps=1_000_000,
    learning_starts=25_000,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    tau=0.005,
    policy_frequency=2,
    log_interval=5_000,
    eval_interval=50_000,
    save_interval=50_000,
    save_dir="saved_models_affine",
    log_dir="logs_affine",
    device="cuda",
    seed=42
):
    """
    Train MARL agents with affine formation control
    
    Returns trained agent
    """
    
    # Import here to avoid circular dependencies
    from affine_matd3 import AffineMatD3
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize CSV logging
    train_csv = os.path.join(log_dir, "training.csv")
    eval_csv = os.path.join(log_dir, "evaluation.csv")
    
    with open(train_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'episode', 'reward', 'length', 'success', 
                        'vote_consensus', 'affine_smoothness'])
    
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'mean_reward', 'success_rate', 'mean_formation_error',
                        'vote_consensus', 'rotation_std', 'scale_std'])
    
    # Initialize agent
    print("Initializing Affine MATD3 agent...")
    agent = AffineMatD3(
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        policy_frequency=policy_frequency,
        batch_size=batch_size,
        learning_starts=learning_starts,
        device=device,
        seed=seed
    )
    
    print(f"Training for {total_timesteps:,} steps")
    print(f"Agents: {agent.num_agents}")
    print(f"Observation dims: {agent.obs_dims}")
    print(f"Device: {agent.device}")
    print("-" * 80)
    
    # Training metrics
    episode_rewards = [0.0] * agent.num_agents
    episode_length = 0
    episode_count = 0
    
    recent_rewards = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    recent_vote_consensus = deque(maxlen=100)
    
    # Curriculum schedule
    curriculum = {
        0: (0, 5, 0),
        50_000: (1, 5, 0),
        100_000: (2, 5, 0),
        150_000: (3, 5, 5),
        250_000: (4, 5, 10),
        400_000: (5, 5, 15)
    }
    
    current_difficulty = (0, 5, 0)
    
    # Initialize environment
    obs_dict, _ = env.reset()
    obs = [obs_dict[agent_name] for agent_name in agent.agents]
    
    # Create evaluation environment
    from affine_formation_env import AffineFormationEnv
    eval_env = AffineFormationEnv(
        num_agents=env.num_agents,
        formation_distance=env.formation_distance,
        gui=False
    )
    
    start_time = time.time()
    
    # === MAIN TRAINING LOOP ===
    for step in range(total_timesteps):
        agent.global_step = step
        
        # === Curriculum Update ===
        if step in curriculum:
            new_walls, new_gaps, new_pillars = curriculum[step]
            print(f"\n*** CURRICULUM LEVEL UP @ Step {step} ***")
            print(f"Obstacles: Walls={new_walls}, Pillars={new_pillars}")
            
            env.set_difficulty(new_walls, new_gaps, new_pillars)
            eval_env.set_difficulty(new_walls, new_gaps, new_pillars)
            current_difficulty = (new_walls, new_gaps, new_pillars)
            
            # Increase exploration
            if step > 0:
                for ou in agent.ou_noises:
                    ou.set_sigma(0.3)
                print(">>> Exploration noise reset to 0.3")
        
        # === Action Selection ===
        if step < learning_starts:
            actions = [env.action_spaces[agent_name].sample() 
                      for agent_name in agent.agents]
        else:
            actions = agent.get_actions(obs, explore=True)
        
        # Get current affine state
        current_affine = env.get_global_affine_state()
        
        # === Environment Step ===
        action_dict = {agent_name: actions[i] 
                      for i, agent_name in enumerate(agent.agents)}
        next_obs_dict, reward_dict, term_dict, trunc_dict, info_dict = env.step(action_dict)
        
        next_obs = [next_obs_dict[agent_name] for agent_name in agent.agents]
        rewards = [reward_dict[agent_name] for agent_name in agent.agents]
        done = any(term_dict.values()) or any(trunc_dict.values())
        
        next_affine = env.get_global_affine_state()
        
        # Track rewards
        for i in range(agent.num_agents):
            episode_rewards[i] += rewards[i]
        episode_length += 1
        
        # === Store Transition ===
        agent.replay_buffer.add(
            obs, next_obs, actions, rewards, done, 
            current_affine, next_affine
        )
        
        obs = next_obs
        
        # === Training Updates ===
        if step >= learning_starts:
            losses = agent.update()
        
        # === Episode End ===
        if done:
            episode_count += 1
            mean_reward = np.mean(episode_rewards)
            success = info_dict.get('agent_0', {}).get('success', False)
            
            # Compute vote consensus
            votes = np.array([actions[i] for i in range(agent.num_agents)])
            vote_var = np.var(votes, axis=0).mean()
            consensus = 1.0 / (1.0 + vote_var)
            
            # Compute affine smoothness
            affine_change = np.abs(next_affine - current_affine).mean()
            
            # Log to CSV
            with open(train_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, episode_count, mean_reward, episode_length,
                               success, consensus, affine_change])
            
            # Track metrics
            recent_rewards.append(mean_reward)
            recent_successes.append(float(success))
            recent_vote_consensus.append(consensus)
            
            # Reset OU noise
            for ou in agent.ou_noises:
                ou.reset()
            
            # Reset environment
            obs_dict, _ = env.reset()
            obs = [obs_dict[agent_name] for agent_name in agent.agents]
            episode_rewards = [0.0] * agent.num_agents
            episode_length = 0
        
        # === Evaluation ===
        if step > learning_starts and step % eval_interval == 0:
            print(f"\n--- Evaluation @ Step {step} ---")
            eval_results = evaluate_affine_policy(
                agent, eval_env, num_episodes=10
            )
            
            # Log evaluation results
            with open(eval_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    eval_results['mean_reward'],
                    eval_results['success_rate'],
                    eval_results['mean_formation_error'],
                    eval_results['vote_consensus'],
                    eval_results['rotation_std'],
                    eval_results['scale_std']
                ])
            
            print(f"Reward: {eval_results['mean_reward']:.2f}")
            print(f"Success: {eval_results['success_rate']*100:.1f}%")
            print(f"Formation Error: {eval_results['mean_formation_error']:.3f}m")
            print(f"Vote Consensus: {eval_results['vote_consensus']:.3f}")
        
        # === Periodic Logging ===
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            sps = (step + 1) / elapsed
            
            print(f"\n{'=' * 80}")
            print(f"Step: {step+1:,} / {total_timesteps:,}")
            print(f"Episodes: {episode_count}")
            print(f"SPS: {sps:.0f}")
            print(f"Time: {elapsed/60:.1f} min")
            
            if len(recent_rewards) > 0:
                print(f"\nRecent 100 Episodes:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
                print(f"  Success Rate: {np.mean(recent_successes)*100:.1f}%")
                print(f"  Vote Consensus: {np.mean(recent_vote_consensus):.3f}")
            
            if step >= learning_starts and losses:
                print(f"\nLosses (Agent 0):")
                print(f"  Critic: {losses.get('agent_0_critic_loss', 0):.4f}")
                if 'agent_0_actor_loss' in losses:
                    print(f"  Actor: {losses.get('agent_0_actor_loss', 0):.4f}")
            
            print(f"{'=' * 80}")
        
        # === Save Model ===
        if (step + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f"checkpoint_step_{step+1}")
            agent.save(save_path)
            print(f"\n✓ Checkpoint saved to {save_path}")
    
    # Final save
    final_path = os.path.join(save_dir, "final_model")
    agent.save(final_path)
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print("Training Complete!")
    print(f"Total Episodes: {episode_count}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Saved to: {final_path}")
    print(f"{'=' * 80}\n")
    
    env.close()
    eval_env.close()
    
    return agent


def evaluate_affine_policy(agent, eval_env, num_episodes=10):
    """
    Evaluate learned affine formation policy
    
    Returns metrics for Q1 paper
    """
    eval_rewards = []
    eval_successes = []
    eval_formation_errors = []
    eval_rotations = []
    eval_scales = []
    all_vote_consensus = []
    
    for ep in range(num_episodes):
        obs_dict, _ = eval_env.reset(seed=1000 + ep)
        obs = [obs_dict[a] for a in agent.agents]
        done = False
        ep_reward = [0.0] * agent.num_agents
        
        ep_formation_errors = []
        ep_rotations = []
        ep_scales = []
        ep_votes = []
        
        while not done:
            # Get actions WITHOUT exploration
            actions = agent.get_actions(obs, explore=False)
            ep_votes.append(actions)
            
            action_dict = {a: actions[i] for i, a in enumerate(agent.agents)}
            next_obs_dict, r_dict, term, trunc, info = eval_env.step(action_dict)
            
            # Track affine parameters
            affine_state = eval_env.get_global_affine_state()
            ep_rotations.append(affine_state[0])
            ep_scales.append(affine_state[1])
            
            # Track formation error
            formation_error = 0
            for a in agent.agents:
                pos = eval_env.agent_positions[a][:2]
                # Get affine-transformed target
                agent_idx = agent.agents.index(a)
                c, s = np.cos(affine_state[0]), np.sin(affine_state[0])
                A = affine_state[1] * np.array([[c, -s], [s, c]])
                target = A @ eval_env.r_nominal[agent_idx] + eval_env.formation_centroid
                formation_error += np.linalg.norm(target - pos)
            
            ep_formation_errors.append(formation_error / agent.num_agents)
            
            obs = [next_obs_dict[a] for a in agent.agents]
            done = any(term.values()) or any(trunc.values())
            
            for i, a in enumerate(agent.agents):
                ep_reward[i] += r_dict[a]
        
        # Compute episode metrics
        eval_rewards.append(np.mean(ep_reward))
        eval_successes.append(info.get('agent_0', {}).get('success', False))
        eval_formation_errors.append(np.mean(ep_formation_errors))
        eval_rotations.extend(ep_rotations)
        eval_scales.extend(ep_scales)
        
        # Compute vote consensus
        if ep_votes:
            votes_array = np.array(ep_votes)
            vote_var = np.var(votes_array, axis=1).mean()
            all_vote_consensus.append(1.0 / (1.0 + vote_var))
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'success_rate': np.mean(eval_successes),
        'mean_formation_error': np.mean(eval_formation_errors),
        'vote_consensus': np.mean(all_vote_consensus),
        'rotation_std': np.std(eval_rotations),
        'scale_std': np.std(eval_scales)
    }


def test_affine_agent(env, model_path, num_episodes=10, device="cuda"):
    """Test trained agent"""
    from affine_matd3 import AffineMatD3
    
    print(f"Loading model from {model_path}...")
    agent = AffineMatD3(env=env, device=device)
    agent.load(model_path)
    
    print(f"Testing for {num_episodes} episodes...")
    
    results = evaluate_affine_policy(agent, env, num_episodes)
    
    print("\n" + "=" * 80)
    print("Test Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Formation Error: {results['mean_formation_error']:.3f}m")
    print(f"Vote Consensus: {results['vote_consensus']:.3f}")
    print(f"Rotation Std: {results['rotation_std']:.3f} rad")
    print(f"Scale Std: {results['scale_std']:.3f}")
    print("=" * 80 + "\n")
    
    env.close()
    return results


if __name__ == "__main__":
    from affine_formation_env import AffineFormationEnv
    
    # Configuration
    TRAIN = True
    TEST = False
    
    NUM_AGENTS = 3
    FORMATION_DISTANCE = 3.0
    GUI = False
    
    TOTAL_TIMESTEPS = 1_000_000
    LEARNING_STARTS = 25_000
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    if TRAIN:
        print("Creating training environment...")
        env = AffineFormationEnv(
            num_agents=NUM_AGENTS,
            formation_distance=FORMATION_DISTANCE,
            gui=GUI
        )
        env.set_difficulty(0, 5, 0)
        
        # Train
        agent = train_affine_matd3(
            env=env,
            total_timesteps=TOTAL_TIMESTEPS,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            seed=SEED
        )
    
    if TEST:
        print("\nCreating test environment...")
        test_env = AffineFormationEnv(
            num_agents=NUM_AGENTS,
            formation_distance=FORMATION_DISTANCE,
            gui=True
        )
        test_env.set_difficulty(5, 5, 15)
        
        # Test
        test_affine_agent(
            env=test_env,
            model_path="saved_models_affine/final_model",
            num_episodes=10,
            device=DEVICE
        )

"""
Training script for MATD3 with Multi-Agent AOATTS Environment
"""

import argparse
import numpy as np
from multi_agent_aoatts import MultiAgentAOATTS
from matd3 import MATD3


def parse_args():
    parser = argparse.ArgumentParser(description="Train MATD3 on Multi-Agent AOATTS")
    
    # Environment parameters
    parser.add_argument("--num-agents", type=int, default=3,
                       help="Number of agents")
    parser.add_argument("--formation-type", type=str, default="triangle",
                       choices=["triangle", "line", "square"],
                       help="Formation type")
    parser.add_argument("--formation-radius", type=float, default=20.0,
                       help="Formation radius")
    parser.add_argument("--gui", action="store_true",
                       help="Use PyBullet GUI")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95,
                       help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01,
                       help="Target network soft update rate")
    parser.add_argument("--policy-noise", type=float, default=0.2,
                       help="Target policy smoothing noise")
    parser.add_argument("--noise-clip", type=float, default=0.5,
                       help="Target policy noise clip")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
                       help="Exploration noise")
    parser.add_argument("--policy-frequency", type=int, default=2,
                       help="Policy update frequency")
    parser.add_argument("--buffer-size", type=int, default=1000000,
                       help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size")
    parser.add_argument("--learning-starts", type=int, default=25000,
                       help="Steps before training starts")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    
    # Logging and saving
    parser.add_argument("--log-interval", type=int, default=10000,
                       help="Logging interval")
    parser.add_argument("--save-path", type=str, default="./models_matd3",
                       help="Path to save models")
    parser.add_argument("--use-tensorboard", action="store_true",
                       help="Use tensorboard logging")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Training MATD3 on Multi-Agent AOATTS Environment")
    print("=" * 80)
    print(f"Number of agents: {args.num_agents}")
    print(f"Formation type: {args.formation_type}")
    print(f"Formation radius: {args.formation_radius}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Create environment
    env = MultiAgentAOATTS(
        num_agents=args.num_agents,
        formation_type=args.formation_type,
        formation_radius=args.formation_radius,
        gui=args.gui,
    )
    
    # Initialize MATD3
    matd3 = MATD3(
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        exploration_noise=args.exploration_noise,
        policy_frequency=args.policy_frequency,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        hidden_dim=args.hidden_dim,
        device=args.device,
        seed=args.seed
    )
    
    print("\nStarting training...")
    print("=" * 80)
    
    # Train
    matd3.train(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_path=args.save_path,
        use_tensorboard=args.use_tensorboard
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Test the trained agents
    print("\nTesting trained agents...")
    mean_rewards = matd3.test(num_episodes=10, verbose=True)
    
    print("\nFinal Test Results:")
    for i, reward in enumerate(mean_rewards):
        print(f"  Agent {i}: {reward:.2f}")
    print(f"  Overall Mean: {np.mean(mean_rewards):.2f}")


if __name__ == "__main__":
    main()
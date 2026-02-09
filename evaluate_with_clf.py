"""
Evaluation script for UAV navigation with CLF-QP filter
"""

from typing import Any, ClassVar, Optional, TypeVar, Union
import os
import time
import numpy as np
import torch as th
from gymnasium import spaces

from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy

from enums import ObservationType, ActionType
from AOATTS_CLF import AOATTS_CLF
from clf_qp_filter import create_clf_filter

import pandas as pd

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.htmlg

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`td3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        grad_clip: float = 0.5,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.grad_clip = grad_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip)
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []


def evaluate_with_clf_filter(
    model_path: str,
    eval_steps: int = 100,
    use_clf: bool = True,
    filter_type: str = "adaptive",
    gui: bool = True,
    output_file: str = "test_info_clf.csv"
):
    """
    Evaluate the TD3 agent with CLF-QP filter.
    
    Args:
        model_path: Path to saved TD3 model
        eval_steps: Number of evaluation episodes
        use_clf: Whether to use CLF filter
        filter_type: "standard" or "adaptive"
        gui: Enable PyBullet visualization
        output_file: Output CSV filename
    """
    
    # Create CLF filter
    clf_filter = None
    if use_clf:
        clf_filter = create_clf_filter(
            agent_vel_max=2.0,
            filter_type=filter_type,
            lambda_clf=1.0,
            gamma_clf=0.5,
            gamma_min=0.1,
            gamma_max=0.8,
            distance_threshold=10.0
        )
        print(f"CLF Filter enabled: {filter_type}")
    else:
        print("CLF Filter disabled")
    
    # Create environment with CLF filter
    env = AOATTS_CLF(
        gui=gui,
        use_clf_filter=use_clf,
        clf_filter=clf_filter
    )
    
    # Load model
    model = TD3.load(model_path, env=env)
    print(f"Model loaded from {model_path}")
    
    # Statistics
    success = 0
    collision = 0
    out_of_bound = 0
    timeout = 0
    
    records = []
    clf_stats = {
        'qp_success_count': 0,
        'clf_satisfied_count': 0,
        'total_action_deviation': 0.0,
        'total_V_values': [],
        'total_dV_dt_values': []
    }
    
    print(f"\nStarting evaluation for {eval_steps} episodes...")
    
    for i in range(eval_steps):
        obs, info = env.reset()
        done, truncated = False, False
        step_count = 0
        episode_V_values = []
        episode_dV_dt_values = []
        episode_action_deviation = 0.0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Collect CLF statistics
            if use_clf and 'clf_info' in info:
                clf_info = info['clf_info']
                if clf_info['qp_success']:
                    clf_stats['qp_success_count'] += 1
                if clf_info['clf_satisfied']:
                    clf_stats['clf_satisfied_count'] += 1
                clf_stats['total_action_deviation'] += clf_info['action_deviation']
                episode_V_values.append(clf_info['V_current'])
                episode_dV_dt_values.append(clf_info['dV_dt'])
                episode_action_deviation += clf_info['action_deviation']
        
        # Record episode results
        print(f"Episode {i+1}/{eval_steps}: {info}")
        
        episode_record = {
            'success': info.get('success', False),
            'termination': info.get('termination', 'unknown'),
            'length': step_count
        }
        
        # Add CLF statistics for this episode
        if use_clf and len(episode_V_values) > 0:
            episode_record['avg_V'] = np.mean(episode_V_values)
            episode_record['final_V'] = episode_V_values[-1]
            episode_record['avg_dV_dt'] = np.mean(episode_dV_dt_values)
            episode_record['avg_action_deviation'] = episode_action_deviation / step_count
            clf_stats['total_V_values'].extend(episode_V_values)
            clf_stats['total_dV_dt_values'].extend(episode_dV_dt_values)
        
        records.append(episode_record)
        
        # Count termination types
        if info.get('termination') == "collision":
            collision += 1
        elif info.get('termination') == "out of bound":
            out_of_bound += 1
        elif info.get('termination') == "timeout":
            timeout += 1
        
        if done:
            success += 1
    
    # Calculate rates
    success_rate = success / eval_steps
    collision_rate = collision / eval_steps
    out_of_bound_rate = out_of_bound / eval_steps
    timeout_rate = timeout / eval_steps
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Success rate: {success_rate:.2%} ({success}/{eval_steps})")
    print(f"Collision rate: {collision_rate:.2%} ({collision}/{eval_steps})")
    print(f"Out of bound rate: {out_of_bound_rate:.2%} ({out_of_bound}/{eval_steps})")
    print(f"Timeout rate: {timeout_rate:.2%} ({timeout}/{eval_steps})")
    
    if use_clf:
        total_steps = sum(r['length'] for r in records)
        print("\n" + "-"*60)
        print("CLF FILTER STATISTICS")
        print("-"*60)
        print(f"QP solver success rate: {clf_stats['qp_success_count']/total_steps:.2%}")
        print(f"CLF constraint satisfied: {clf_stats['clf_satisfied_count']/total_steps:.2%}")
        print(f"Average action deviation: {clf_stats['total_action_deviation']/total_steps:.4f}")
        print(f"Average V(x): {np.mean(clf_stats['total_V_values']):.4f}")
        print(f"Average dV/dt: {np.mean(clf_stats['total_dV_dt_values']):.4f}")
    
    print("="*60)
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'out_of_bound_rate': out_of_bound_rate,
        'timeout_rate': timeout_rate,
        'records': records,
        'clf_stats': clf_stats if use_clf else None
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate UAV navigation with CLF-QP filter')
    parser.add_argument('--model', type=str, default='td3_aoatt_new963.zip',
                        help='Path to trained TD3 model')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-clf', action='store_true',
                        help='Disable CLF filter (baseline)')
    parser.add_argument('--filter-type', type=str, default='adaptive',
                        choices=['standard', 'adaptive'],
                        help='Type of CLF filter')
    parser.add_argument('--gui', action='store_true',
                        help='Enable PyBullet GUI')
    parser.add_argument('--output', type=str, default='test_info_clf.csv',
                        help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_with_clf_filter(
        model_path=args.model,
        eval_steps=args.steps,
        use_clf=not args.no_clf,
        filter_type=args.filter_type,
        gui=args.gui,
        output_file=args.output
    )

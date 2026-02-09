"""
Evaluation script for Multi-Rate UAV navigation
RL at 1 Hz, CLF-QP at 10 Hz, Physics at 100 Hz
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Import the TD3 class and environment
from evaluate_with_clf import TD3
from AOATTS_CLF_MultiRate import AOATTS_CLF_MultiRate
from clf_qp_filter import create_clf_filter


def evaluate_multirate(
    model_path: str,
    eval_steps: int = 100,
    use_clf: bool = True,
    filter_type: str = "adaptive",
    rl_frequency: float = 1.0,
    qp_frequency: float = 10.0,
    gui: bool = True,
    output_file: str = "test_info_multirate.csv"
):
    """
    Evaluate TD3 agent with multi-rate CLF-QP filter.
    
    Args:
        model_path: Path to saved TD3 model
        eval_steps: Number of evaluation episodes
        use_clf: Whether to use CLF filter
        filter_type: "standard" or "adaptive"
        rl_frequency: RL policy update frequency (Hz)
        qp_frequency: QP filter update frequency (Hz)
        gui: Enable PyBullet visualization
        output_file: Output CSV filename
    """
    
    print("\n" + "="*80)
    print("MULTI-RATE UAV NAVIGATION EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Episodes: {eval_steps}")
    print(f"RL Frequency: {rl_frequency} Hz")
    print(f"QP Frequency: {qp_frequency} Hz")
    print(f"CLF Filter: {'Enabled (' + filter_type + ')' if use_clf else 'Disabled'}")
    print("="*80 + "\n")
    
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
        print(f"✓ CLF Filter created: {filter_type}")
    
    # Create environment with multi-rate control
    env = AOATTS_CLF_MultiRate(
        gui=gui,
        use_clf_filter=use_clf,
        clf_filter=clf_filter,
        rl_frequency=rl_frequency,
        qp_frequency=qp_frequency,
        physics_frequency=100.0
    )
    
    # Load model
    model = TD3.load(model_path, env=env)
    print(f"✓ Model loaded from {model_path}\n")
    
    # Statistics
    success = 0
    collision = 0
    out_of_bound = 0
    timeout = 0
    
    records = []
    overall_qp_stats = {
        'total_qp_calls': 0,
        'total_qp_success': 0,
        'total_action_deviation': 0.0,
        'episode_V_values': [],
        'episode_dV_dt_values': []
    }
    
    print(f"Starting evaluation for {eval_steps} episodes...")
    print("-" * 80)
    
    for i in range(eval_steps):
        obs, info = env.reset()
        done, truncated = False, False
        step_count = 0
        episode_V_values = []
        episode_dV_dt_values = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            
            # Collect CLF statistics
            if use_clf and 'clf_info' in info:
                clf_info = info['clf_info']
                episode_V_values.append(clf_info['V_current'])
                episode_dV_dt_values.append(clf_info['dV_dt'])
        
        # Record episode results
        episode_record = {
            'episode': i + 1,
            'success': info.get('success', False),
            'termination': info.get('termination', 'unknown'),
            'length': step_count,
            'rl_steps': step_count,
        }
        
        # Add multi-rate QP statistics
        if use_clf:
            qp_calls = info.get('qp_calls_this_episode', 0)
            qp_success_rate = info.get('qp_success_rate', 0)
            avg_deviation = info.get('avg_action_deviation', 0)
            
            episode_record['qp_calls'] = qp_calls
            episode_record['qp_steps'] = qp_calls
            episode_record['qp_success_rate'] = qp_success_rate
            episode_record['avg_action_deviation'] = avg_deviation
            
            overall_qp_stats['total_qp_calls'] += qp_calls
            overall_qp_stats['total_qp_success'] += int(qp_calls * qp_success_rate)
            overall_qp_stats['total_action_deviation'] += avg_deviation * qp_calls
            
            if len(episode_V_values) > 0:
                episode_record['avg_V'] = np.mean(episode_V_values)
                episode_record['final_V'] = episode_V_values[-1]
                episode_record['avg_dV_dt'] = np.mean(episode_dV_dt_values)
                overall_qp_stats['episode_V_values'].extend(episode_V_values)
                overall_qp_stats['episode_dV_dt_values'].extend(episode_dV_dt_values)
        
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
        
        # Print progress
        status = "✓" if done else "✗"
        print(f"{status} Episode {i+1:3d}/{eval_steps}: "
              f"{'SUCCESS' if done else info.get('termination', 'FAILED'):12s} "
              f"| Steps: {step_count:3d} "
              f"| QP calls: {episode_record.get('qp_calls', 0):4d}" if use_clf else "")
    
    # Calculate rates
    success_rate = success / eval_steps
    collision_rate = collision / eval_steps
    out_of_bound_rate = out_of_bound / eval_steps
    timeout_rate = timeout / eval_steps
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Success rate:        {success_rate:6.2%}  ({success}/{eval_steps})")
    print(f"Collision rate:      {collision_rate:6.2%}  ({collision}/{eval_steps})")
    print(f"Out of bound rate:   {out_of_bound_rate:6.2%}  ({out_of_bound}/{eval_steps})")
    print(f"Timeout rate:        {timeout_rate:6.2%}  ({timeout}/{eval_steps})")
    
    # Print multi-rate statistics
    if use_clf and overall_qp_stats['total_qp_calls'] > 0:
        print("\n" + "-"*80)
        print("MULTI-RATE CLF-QP STATISTICS")
        print("-"*80)
        print(f"Total RL steps:      {sum(r['rl_steps'] for r in records):,}")
        print(f"Total QP updates:    {overall_qp_stats['total_qp_calls']:,}")
        print(f"QP/RL ratio:         {overall_qp_stats['total_qp_calls'] / sum(r['rl_steps'] for r in records):.1f}x")
        print(f"QP solver success:   {overall_qp_stats['total_qp_success'] / overall_qp_stats['total_qp_calls']:6.2%}")
        print(f"Avg action deviation: {overall_qp_stats['total_action_deviation'] / overall_qp_stats['total_qp_calls']:.4f}")
        print(f"Avg V(x):            {np.mean(overall_qp_stats['episode_V_values']):.4f}")
        print(f"Avg dV/dt:           {np.mean(overall_qp_stats['episode_dV_dt_values']):.4f}")
        
        # Convergence analysis
        V_initial = np.mean([r.get('avg_V', 0) for r in records[:10] if 'avg_V' in r])
        V_final = np.mean([r.get('final_V', 0) for r in records if 'final_V' in r])
        print(f"\nConvergence Analysis:")
        print(f"  Initial avg V(x):  {V_initial:.4f}")
        print(f"  Final avg V(x):    {V_final:.4f}")
        print(f"  Reduction:         {(V_initial - V_final) / V_initial * 100:.1f}%")
    
    print("="*80 + "\n")
    
    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"✓ Results saved to {output_file}\n")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'out_of_bound_rate': out_of_bound_rate,
        'timeout_rate': timeout_rate,
        'records': records,
        'qp_stats': overall_qp_stats if use_clf else None
    }


def compare_frequencies(
    model_path: str,
    eval_steps: int = 50,
    output_prefix: str = "freq_comparison"
):
    """
    Compare different QP frequencies.
    
    Tests: 1Hz, 5Hz, 10Hz, 20Hz, 50Hz
    """
    frequencies = [1.0, 5.0, 10.0, 20.0, 50.0]
    results = {}
    
    print("\n" + "="*80)
    print("FREQUENCY COMPARISON STUDY")
    print("="*80)
    print(f"Testing QP frequencies: {frequencies}")
    print("="*80 + "\n")
    
    for qp_freq in frequencies:
        print(f"\n{'='*80}")
        print(f"Testing QP Frequency: {qp_freq} Hz")
        print(f"{'='*80}")
        
        output_file = f"{output_prefix}_{int(qp_freq)}hz.csv"
        
        results[qp_freq] = evaluate_multirate(
            model_path=model_path,
            eval_steps=eval_steps,
            use_clf=True,
            filter_type="adaptive",
            rl_frequency=1.0,
            qp_frequency=qp_freq,
            gui=True,
            output_file=output_file
        )
    
    # Summary comparison
    print("\n" + "="*80)
    print("FREQUENCY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'QP Freq (Hz)':<15} {'Success':<10} {'Collision':<12} {'Timeout':<10} {'QP/RL Ratio':<12}")
    print("-"*80)
    
    for freq in frequencies:
        r = results[freq]
        qp_ratio = r['qp_stats']['total_qp_calls'] / sum(rec['rl_steps'] for rec in r['records']) if r['qp_stats'] else 0
        print(f"{freq:<15.1f} {r['success_rate']*100:>7.2f}%  {r['collision_rate']*100:>9.2f}%  "
              f"{r['timeout_rate']*100:>7.2f}%  {qp_ratio:>10.1f}x")
    
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate UAV navigation with multi-rate CLF-QP filter'
    )
    parser.add_argument('--model', type=str, default='td3_aoatt_new963.zip',
                        help='Path to trained TD3 model')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-clf', action='store_true',
                        help='Disable CLF filter (baseline)')
    parser.add_argument('--filter-type', type=str, default='adaptive',
                        choices=['standard', 'adaptive'],
                        help='Type of CLF filter')
    parser.add_argument('--rl-freq', type=float, default=1.0,
                        help='RL policy frequency in Hz')
    parser.add_argument('--qp-freq', type=float, default=10.0,
                        help='QP filter frequency in Hz')
    parser.add_argument('--gui', action='store_true',
                        help='Enable PyBullet GUI')
    parser.add_argument('--output', type=str, default='test_info_multirate.csv',
                        help='Output CSV filename')
    parser.add_argument('--compare-frequencies', action='store_true',
                        help='Run frequency comparison study')
    
    args = parser.parse_args()
    
    if args.compare_frequencies:
        # Run frequency comparison
        compare_frequencies(
            model_path=args.model,
            eval_steps=args.steps,
            output_prefix='freq_comparison'
        )
    else:
        # Run single evaluation
        results = evaluate_multirate(
            model_path=args.model,
            eval_steps=args.steps,
            use_clf=not args.no_clf,
            filter_type=args.filter_type,
            rl_frequency=args.rl_freq,
            qp_frequency=args.qp_freq,
            gui=args.gui,
            output_file=args.output
        )

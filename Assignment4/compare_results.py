"""
Utility script to compare performance of different algorithms
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def load_test_results(algorithm, env_name):
    """Load test results from CSV file"""
    env_map = {
        'lunarlander': 'LunarLander-v3',
        'carracing': 'CarRacing-v3'
    }
    
    filename = f"results/{algorithm.upper()}_{env_map[env_name]}_test_results.csv"
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    
    df = pd.read_csv(filename)
    return df


def plot_comparison(env_name='lunarlander', algorithms=['sac', 'ppo', 'td3']):
    """Plot comparison of different algorithms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Algorithm Comparison on {env_name.upper()}', fontsize=16)
    
    colors = {'sac': 'blue', 'ppo': 'green', 'td3': 'red'}
    
    all_data = {}
    
    # Load data for all algorithms
    for algo in algorithms:
        df = load_test_results(algo, env_name)
        if df is not None:
            all_data[algo] = df
    
    if not all_data:
        print("No data found for any algorithm!")
        return
    
    # Plot 1: Episode Rewards over time
    ax1 = axes[0, 0]
    for algo, df in all_data.items():
        ax1.plot(df['Episode'], df['Reward'], alpha=0.3, color=colors[algo])
        # Add smoothed line
        window = 10
        smoothed = df['Reward'].rolling(window=window, min_periods=1).mean()
        ax1.plot(df['Episode'], smoothed, label=algo.upper(), color=colors[algo], linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards (smoothed)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward Distribution (Box Plot)
    ax2 = axes[0, 1]
    data_for_box = [df['Reward'].values for df in all_data.values()]
    box = ax2.boxplot(data_for_box, labels=[a.upper() for a in all_data.keys()],
                       patch_artist=True)
    
    for patch, algo in zip(box['boxes'], all_data.keys()):
        patch.set_facecolor(colors[algo])
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cumulative Reward
    ax3 = axes[1, 0]
    for algo, df in all_data.items():
        cumulative = df['Reward'].cumsum()
        ax3.plot(df['Episode'], cumulative, label=algo.upper(), color=colors[algo], linewidth=2)
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Reward Over Episodes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics Comparison (Bar Plot)
    ax4 = axes[1, 1]
    
    stats = {}
    for algo, df in all_data.items():
        stats[algo] = {
            'Mean': df['Reward'].mean(),
            'Std': df['Reward'].std(),
            'Max': df['Reward'].max(),
            'Min': df['Reward'].min()
        }
    
    x = np.arange(len(all_data))
    width = 0.2
    
    means = [stats[algo]['Mean'] for algo in all_data.keys()]
    stds = [stats[algo]['Std'] for algo in all_data.keys()]
    
    bars = ax4.bar(x, means, width, label='Mean Reward', 
                    color=[colors[algo] for algo in all_data.keys()], alpha=0.7)
    ax4.errorbar(x, means, yerr=stds, fmt='none', color='black', capsize=5)
    
    ax4.set_ylabel('Reward')
    ax4.set_title('Mean Reward Comparison (with std)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([algo.upper() for algo in all_data.keys()])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/comparison_{env_name}.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to results/comparison_{env_name}.png")
    
    plt.show()


def print_statistics(env_name='lunarlander', algorithms=['sac', 'ppo', 'td3']):
    """Print detailed statistics for all algorithms"""
    
    print("\n" + "="*80)
    print(f"Algorithm Comparison on {env_name.upper()}")
    print("="*80)
    
    for algo in algorithms:
        df = load_test_results(algo, env_name)
        
        if df is None:
            print(f"\n{algo.upper()}: No data available")
            continue
        
        rewards = df['Reward'].values
        lengths = df['Length'].values
        
        print(f"\n{algo.upper()}:")
        print("-" * 40)
        print(f"  Mean Reward:     {rewards.mean():10.2f} ± {rewards.std():.2f}")
        print(f"  Median Reward:   {np.median(rewards):10.2f}")
        print(f"  Min Reward:      {rewards.min():10.2f}")
        print(f"  Max Reward:      {rewards.max():10.2f}")
        print(f"  Mean Length:     {lengths.mean():10.2f}")
        print(f"  Success Rate:    {(rewards > 200).mean()*100:10.1f}%")  # Assuming 200 is success threshold
    
    print("\n" + "="*80)


def generate_summary_table(env_name='lunarlander', algorithms=['sac', 'ppo', 'td3']):
    """Generate a summary table comparing all algorithms"""
    
    summary = []
    
    for algo in algorithms:
        df = load_test_results(algo, env_name)
        
        if df is None:
            continue
        
        rewards = df['Reward'].values
        lengths = df['Length'].values
        
        summary.append({
            'Algorithm': algo.upper(),
            'Mean Reward': f"{rewards.mean():.2f}",
            'Std Reward': f"{rewards.std():.2f}",
            'Max Reward': f"{rewards.max():.2f}",
            'Min Reward': f"{rewards.min():.2f}",
            'Mean Length': f"{lengths.mean():.2f}",
            'Success Rate': f"{(rewards > 200).mean()*100:.1f}%"
        })
    
    if summary:
        summary_df = pd.DataFrame(summary)
        
        # Save to CSV
        filename = f"results/comparison_summary_{env_name}.csv"
        summary_df.to_csv(filename, index=False)
        print(f"\nSummary table saved to {filename}")
        
        # Print table
        print("\n" + summary_df.to_string(index=False))
    else:
        print("No data available for summary table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare algorithm performance')
    parser.add_argument('--env', type=str, default='lunarlander',
                        choices=['lunarlander', 'carracing'],
                        help='Environment to compare')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['sac', 'ppo', 'td3'],
                        help='Algorithms to compare')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    
    args = parser.parse_args()
    
    # Print statistics
    print_statistics(args.env, args.algorithms)
    
    # Generate summary table
    generate_summary_table(args.env, args.algorithms)
    
    # Plot comparison
    if not args.no_plot:
        plot_comparison(args.env, args.algorithms)

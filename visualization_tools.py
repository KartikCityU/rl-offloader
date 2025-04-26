import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Import required classes from edge_offloading_rl
from edge_offloading_rl import Task, EdgeDevice, CloudServer, NetworkCondition, EdgeEnvironment
from edge_offloading_rl import DQNAgent, HeuristicAgent

def create_custom_colormap():
    """Create a custom colormap for visualizations."""
    colors = [(0.8, 0.2, 0.2), (0.95, 0.95, 0.0), (0.2, 0.8, 0.2)]
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

def plot_decision_boundaries(agent, env, grid_size=20):
    """
    Plot decision boundaries of an agent in a 2D state space.
    
    Args:
        agent: The agent (RL or heuristic)
        env: The environment
        grid_size: Resolution of the grid
    """
    # We'll visualize decisions based on task complexity and network latency
    # while keeping other state variables fixed
    
    # Fixed values for other state variables
    data_size = 0.5  # Normalized
    bandwidth = 0.5  # Normalized
    energy = 0.8  # 80% battery
    deadline = 0.7  # Normalized
    
    # Create meshgrid for task complexity and latency
    complexity_range = np.linspace(0, 1, grid_size)
    latency_range = np.linspace(0, 1, grid_size)
    complexity_grid, latency_grid = np.meshgrid(complexity_range, latency_range)
    
    # Initialize decision grid
    decision_grid = np.zeros((grid_size, grid_size))
    
    # Get decisions for each point in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            complexity = complexity_grid[i, j]
            latency = latency_grid[i, j]
            
            # Create state
            state = np.array([complexity, data_size, latency, bandwidth, energy, deadline])
            
            # Get action (0: local, 1: offload)
            if isinstance(agent, HeuristicAgent):
                # Create task and network conditions for heuristic agent
                task = Task(
                    task_id=1,
                    complexity=complexity * 1e9,  # Denormalize
                    data_size=data_size * 5000,  # Denormalize
                    deadline=deadline * 10000  # Denormalize
                )
                network_conditions = (latency * 200, bandwidth * 10000)  # Denormalize
                action = agent.act(
                    state, env.edge_device, env.cloud_server, task, network_conditions
                )
            else:
                # RL agent
                action = agent.act(state)
            
            decision_grid[i, j] = action
    
    # Plot decision boundaries
    plt.figure(figsize=(10, 8))
    cmap = create_custom_colormap()
    plt.imshow(decision_grid, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    plt.colorbar(label='Decision (0: Local, 1: Offload)')
    plt.xlabel('Task Complexity (Normalized)')
    plt.ylabel('Network Latency (Normalized)')
    plt.title('Task Offloading Decision Boundaries')
    
    # Add grid lines
    plt.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.savefig('results/decision_boundaries.png', dpi=300)
    plt.close()

def plot_reward_landscape(env, grid_size=20):
    """
    Plot the reward landscape for different decisions.
    
    Args:
        env: The environment
        grid_size: Resolution of the grid
    """
    # We'll visualize rewards based on task complexity and network latency
    # for both local and offloaded execution
    
    # Fixed values for other variables
    data_size = 0.5  # Normalized
    bandwidth = 0.5  # Normalized
    energy = 0.8  # 80% battery
    deadline = 0.7  # Normalized
    
    # Create meshgrid for task complexity and latency
    complexity_range = np.linspace(0, 1, grid_size)
    latency_range = np.linspace(0, 1, grid_size)
    complexity_grid, latency_grid = np.meshgrid(complexity_range, latency_range)
    
    # Initialize reward grids
    local_reward_grid = np.zeros((grid_size, grid_size))
    offload_reward_grid = np.zeros((grid_size, grid_size))
    optimal_reward_grid = np.zeros((grid_size, grid_size))
    optimal_decision_grid = np.zeros((grid_size, grid_size))
    
    # Calculate rewards for each point in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            complexity = complexity_grid[i, j]
            latency = latency_grid[i, j]
            
            # Create task for reward calculation
            task = Task(
                task_id=1,
                complexity=complexity * 1e9,  # Denormalize
                data_size=data_size * 5000,  # Denormalize
                deadline=deadline * 10000  # Denormalize
            )
            
            # Get network conditions
            network_latency = latency * 200  # Denormalize
            network_bandwidth = bandwidth * 10000  # Denormalize
            
            # Calculate rewards for local execution
            local_time, local_energy = env.edge_device.get_local_execution_estimate(task)
            local_deadline_violated = task.deadline and local_time > task.deadline
            local_reward = env.calculate_reward(local_time, local_energy, 0, local_deadline_violated)
            local_reward_grid[i, j] = local_reward
            
            # Calculate rewards for offloaded execution
            offload_time, offload_cost = env.cloud_server.get_cloud_execution_estimate(
                task, network_latency, network_bandwidth
            )
            offload_deadline_violated = task.deadline and offload_time > task.deadline
            offload_reward = env.calculate_reward(offload_time, 0, offload_cost, offload_deadline_violated)
            offload_reward_grid[i, j] = offload_reward
            
            # Determine optimal decision
            if local_reward >= offload_reward:
                optimal_reward_grid[i, j] = local_reward
                optimal_decision_grid[i, j] = 0  # Local
            else:
                optimal_reward_grid[i, j] = offload_reward
                optimal_decision_grid[i, j] = 1  # Offload
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot local execution rewards
    im0 = axes[0, 0].imshow(local_reward_grid, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[0, 0].set_title('Local Execution Rewards')
    axes[0, 0].set_xlabel('Task Complexity (Normalized)')
    axes[0, 0].set_ylabel('Network Latency (Normalized)')
    fig.colorbar(im0, ax=axes[0, 0], label='Reward')
    
    # Plot offloaded execution rewards
    im1 = axes[0, 1].imshow(offload_reward_grid, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[0, 1].set_title('Offloaded Execution Rewards')
    axes[0, 1].set_xlabel('Task Complexity (Normalized)')
    axes[0, 1].set_ylabel('Network Latency (Normalized)')
    fig.colorbar(im1, ax=axes[0, 1], label='Reward')
    
    # Plot optimal rewards
    im2 = axes[1, 0].imshow(optimal_reward_grid, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[1, 0].set_title('Optimal Rewards')
    axes[1, 0].set_xlabel('Task Complexity (Normalized)')
    axes[1, 0].set_ylabel('Network Latency (Normalized)')
    fig.colorbar(im2, ax=axes[1, 0], label='Reward')
    
    # Plot optimal decisions
    cmap = create_custom_colormap()
    im3 = axes[1, 1].imshow(optimal_decision_grid, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    axes[1, 1].set_title('Optimal Decisions (0: Local, 1: Offload)')
    axes[1, 1].set_xlabel('Task Complexity (Normalized)')
    axes[1, 1].set_ylabel('Network Latency (Normalized)')
    fig.colorbar(im3, ax=axes[1, 1], label='Decision')
    
    plt.tight_layout()
    plt.savefig('results/reward_landscape.png', dpi=300)
    plt.close()

def plot_performance_over_time(results_df, window_size=10):
    """
    Plot performance metrics over time with smoothing.
    
    Args:
        results_df: DataFrame with performance metrics
        window_size: Window size for rolling average smoothing
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot execution time
    for agent in results_df['agent'].unique():
        agent_data = results_df[results_df['agent'] == agent]
        # Apply rolling average only to numeric columns for each agent separately
        if len(agent_data) > window_size:
            smoothed = agent_data['execution_time'].rolling(window=window_size, min_periods=1).mean()
            axes[0, 0].plot(agent_data['step'], smoothed, label=agent)
        else:
            axes[0, 0].plot(agent_data['step'], agent_data['execution_time'], label=agent)
    
    axes[0, 0].set_title('Execution Time Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Execution Time (ms)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot energy consumption
    for agent in results_df['agent'].unique():
        agent_data = results_df[results_df['agent'] == agent]
        if len(agent_data) > window_size:
            smoothed = agent_data['energy_consumed'].rolling(window=window_size, min_periods=1).mean()
            axes[0, 1].plot(agent_data['step'], smoothed, label=agent)
        else:
            axes[0, 1].plot(agent_data['step'], agent_data['energy_consumed'], label=agent)
    
    axes[0, 1].set_title('Energy Consumption Over Time')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Energy (J)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot cloud cost
    for agent in results_df['agent'].unique():
        agent_data = results_df[results_df['agent'] == agent]
        if len(agent_data) > window_size:
            smoothed = agent_data['cloud_cost'].rolling(window=window_size, min_periods=1).mean()
            axes[1, 0].plot(agent_data['step'], smoothed, label=agent)
        else:
            axes[1, 0].plot(agent_data['step'], agent_data['cloud_cost'], label=agent)
    
    axes[1, 0].set_title('Cloud Cost Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cost ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot reward
    for agent in results_df['agent'].unique():
        agent_data = results_df[results_df['agent'] == agent]
        if len(agent_data) > window_size:
            smoothed = agent_data['reward'].rolling(window=window_size, min_periods=1).mean()
            axes[1, 1].plot(agent_data['step'], smoothed, label=agent)
        else:
            axes[1, 1].plot(agent_data['step'], agent_data['reward'], label=agent)
    
    axes[1, 1].set_title('Reward Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/performance_over_time.png', dpi=300)
    plt.close()

def plot_offloading_rate(results_df):
    """
    Plot offloading rate by task complexity.
    
    Args:
        results_df: DataFrame with performance metrics
    """
    # Bin task complexity
    bins = np.linspace(0, 1, 6)
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    results_df_copy = results_df.copy()
    results_df_copy['complexity_bin'] = pd.cut(
        results_df_copy['task_complexity'], bins=bins, labels=labels
    )
    
    # Calculate offloading rate by complexity bin for each agent
    offloading_rates = results_df_copy.groupby(['agent', 'complexity_bin'])['action'].mean().reset_index()
    
    # Pivot the data for plotting
    pivot_df = offloading_rates.pivot(index='complexity_bin', columns='agent', values='action')
    
    # Plot offloading rates
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Offloading Rate by Task Complexity')
    plt.xlabel('Task Complexity')
    plt.ylabel('Offloading Rate')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Agent')
    
    plt.tight_layout()
    plt.savefig('results/offloading_rate.png', dpi=300)
    plt.close()

def plot_energy_efficiency(results_df):
    """
    Plot energy efficiency metrics.
    
    Args:
        results_df: DataFrame with performance metrics
    """
    # Calculate energy efficiency (execution time per joule)
    results_df_copy = results_df.copy()
    results_df_copy['energy_efficiency'] = results_df_copy['execution_time'] / (results_df_copy['energy_consumed'] + 0.001)
    
    # Group by agent and whether the task was offloaded
    grouped_df = results_df_copy.groupby(['agent', 'action']).agg({
        'energy_efficiency': 'mean',
        'execution_time': 'mean',
        'energy_consumed': 'mean'
    }).reset_index()
    
    # Replace action values with descriptive labels
    grouped_df['execution'] = grouped_df['action'].map({0: 'Local', 1: 'Offloaded'})
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot energy efficiency
    sns.barplot(x='agent', y='energy_efficiency', hue='execution', data=grouped_df, ax=axes[0])
    axes[0].set_title('Energy Efficiency')
    axes[0].set_xlabel('Agent')
    axes[0].set_ylabel('Execution Time per Joule')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot execution time
    sns.barplot(x='agent', y='execution_time', hue='execution', data=grouped_df, ax=axes[1])
    axes[1].set_title('Execution Time')
    axes[1].set_xlabel('Agent')
    axes[1].set_ylabel('Time (ms)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot energy consumption
    sns.barplot(x='agent', y='energy_consumed', hue='execution', data=grouped_df, ax=axes[2])
    axes[2].set_title('Energy Consumption')
    axes[2].set_xlabel('Agent')
    axes[2].set_ylabel('Energy (J)')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/energy_efficiency.png', dpi=300)
    plt.close()

def plot_deadline_violations(results_df):
    """
    Plot deadline violation rates.
    
    Args:
        results_df: DataFrame with performance metrics
    """
    # Calculate deadline violation rate by agent and network condition
    # Bin network latency
    bins = np.linspace(0, 1, 4)
    labels = ['Low', 'Medium', 'High']
    results_df_copy = results_df.copy()
    results_df_copy['latency_bin'] = pd.cut(
        results_df_copy['network_latency'], bins=bins, labels=labels
    )
    
    # Calculate violation rate
    violation_rates = results_df_copy.groupby(['agent', 'latency_bin'])['deadline_violated'].mean().reset_index()
    
    # Pivot the data for plotting
    pivot_df = violation_rates.pivot(index='latency_bin', columns='agent', values='deadline_violated')
    
    # Plot violation rates
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Deadline Violation Rate by Network Latency')
    plt.xlabel('Network Latency')
    plt.ylabel('Violation Rate')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Agent')
    
    plt.tight_layout()
    plt.savefig('results/deadline_violations.png', dpi=300)
    plt.close()

def create_comprehensive_dashboard(results_df, rl_agent, env):
    """
    Create a comprehensive performance dashboard.
    
    Args:
        results_df: DataFrame with performance metrics
        rl_agent: The trained RL agent
        env: The environment
    """
    # Create a large figure with GridSpec
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1])
    
    # 1. Decision Boundaries (RL Agent)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get decision boundaries for RL agent
    grid_size = 50
    complexity_range = np.linspace(0, 1, grid_size)
    latency_range = np.linspace(0, 1, grid_size)
    complexity_grid, latency_grid = np.meshgrid(complexity_range, latency_range)
    decision_grid = np.zeros((grid_size, grid_size))
    
    # Fixed values for other state variables
    data_size = 0.5
    bandwidth = 0.5
    energy = 0.8
    deadline = 0.7
    
    for i in range(grid_size):
        for j in range(grid_size):
            complexity = complexity_grid[i, j]
            latency = latency_grid[i, j]
            state = np.array([complexity, data_size, latency, bandwidth, energy, deadline])
            action = rl_agent.act(state)
            decision_grid[i, j] = action
    
    cmap = create_custom_colormap()
    im1 = ax1.imshow(decision_grid, cmap=cmap, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
    ax1.set_title('RL Agent Decision Boundaries')
    ax1.set_xlabel('Task Complexity (Normalized)')
    ax1.set_ylabel('Network Latency (Normalized)')
    fig.colorbar(im1, ax=ax1, label='Decision (0: Local, 1: Offload)')
    
    # 2. Performance Comparison (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate average metrics by agent
    avg_metrics = results_df.groupby('agent').agg({
        'execution_time': 'mean',
        'energy_consumed': 'mean',
        'cloud_cost': 'mean',
        'reward': 'mean',
        'deadline_violated': 'mean'
    }).reset_index()
    
    # Normalize metrics for fair comparison
    metrics_to_normalize = ['execution_time', 'energy_consumed', 'cloud_cost', 'deadline_violated']
    for metric in metrics_to_normalize:
        max_val = avg_metrics[metric].max()
        if max_val > 0:
            avg_metrics[f'{metric}_norm'] = avg_metrics[metric] / max_val
    
    # Invert normalized metrics so lower is better
    for metric in metrics_to_normalize:
        avg_metrics[f'{metric}_norm'] = 1 - avg_metrics[f'{metric}_norm']
    
    # Calculate overall score (higher is better)
    weights = {
        'execution_time_norm': 0.3,
        'energy_consumed_norm': 0.3,
        'cloud_cost_norm': 0.2,
        'deadline_violated_norm': 0.2
    }
    
    avg_metrics['overall_score'] = sum(
        avg_metrics[metric] * weight for metric, weight in weights.items()
    )
    
    # Plot overall score
    sns.barplot(x='agent', y='overall_score', data=avg_metrics, ax=ax2)
    ax2.set_title('Overall Performance Score (Higher is Better)')
    ax2.set_xlabel('Agent')
    ax2.set_ylabel('Score')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Offloading Rate by Task Complexity
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Bin task complexity
    bins = np.linspace(0, 1, 6)
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    results_df_copy = results_df.copy()
    results_df_copy['complexity_bin'] = pd.cut(
        results_df_copy['task_complexity'], bins=bins, labels=labels
    )
    
    # Calculate offloading rate by complexity bin for each agent
    offloading_rates = results_df_copy.groupby(['agent', 'complexity_bin'])['action'].mean().reset_index()
    
    # Pivot the data for plotting
    pivot_df = offloading_rates.pivot(index='complexity_bin', columns='agent', values='action')
    
    # Plot offloading rates
    pivot_df.plot(kind='bar', ax=ax3)
    ax3.set_title('Offloading Rate by Task Complexity')
    ax3.set_xlabel('Task Complexity')
    ax3.set_ylabel('Offloading Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(title='Agent')
    
    # 4. Energy Consumption Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Apply rolling average smoothing
    window_size = 20
    
    # Plot energy consumption
    for agent in results_df['agent'].unique():
        agent_data = results_df[results_df['agent'] == agent]
        if len(agent_data) > window_size:
            smoothed = agent_data['energy_consumed'].rolling(window=window_size, min_periods=1).mean()
            ax4.plot(agent_data['step'], smoothed, label=agent)
        else:
            ax4.plot(agent_data['step'], agent_data['energy_consumed'], label=agent)
    
    ax4.set_title('Energy Consumption Over Time')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Energy (J)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Execution Time Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    
    sns.boxplot(x='agent', y='execution_time', data=results_df, ax=ax5)
    ax5.set_title('Execution Time Distribution')
    ax5.set_xlabel('Agent')
    ax5.set_ylabel('Execution Time (ms)')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Deadline Violation Rate by Network Condition
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Bin network latency
    bins = np.linspace(0, 1, 4)
    labels = ['Low', 'Medium', 'High']
    results_df_copy = results_df.copy()
    results_df_copy['latency_bin'] = pd.cut(
        results_df_copy['network_latency'], bins=bins, labels=labels
    )
    
    # Calculate violation rate
    violation_rates = results_df_copy.groupby(['agent', 'latency_bin'])['deadline_violated'].mean().reset_index()
    
    # Pivot the data for plotting
    pivot_df = violation_rates.pivot(index='latency_bin', columns='agent', values='deadline_violated')
    
    # Plot violation rates
    pivot_df.plot(kind='bar', ax=ax6)
    ax6.set_title('Deadline Violation Rate by Network Latency')
    ax6.set_xlabel('Network Latency')
    ax6.set_ylabel('Violation Rate')
    ax6.set_ylim(0, 1)
    ax6.grid(axis='y', alpha=0.3)
    ax6.legend(title='Agent')
    
    # 7. Cloud Cost Analysis
    ax7 = fig.add_subplot(gs[3, 0])
    
    # Calculate average cloud cost by agent
    cloud_cost_by_agent = results_df.groupby('agent')['cloud_cost'].mean().reset_index()
    
    sns.barplot(x='agent', y='cloud_cost', data=cloud_cost_by_agent, ax=ax7)
    ax7.set_title('Average Cloud Cost')
    ax7.set_xlabel('Agent')
    ax7.set_ylabel('Cost ($)')
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Energy Efficiency (Performance per Joule)
    ax8 = fig.add_subplot(gs[3, 1])
    
    # Calculate performance per joule (add small epsilon to avoid division by zero)
    results_df_copy = results_df.copy()
    results_df_copy['perf_per_joule'] = 1000 / (results_df_copy['execution_time'] * (results_df_copy['energy_consumed'] + 0.001))
    perf_by_agent = results_df_copy.groupby('agent')['perf_per_joule'].mean().reset_index()
    
    sns.barplot(x='agent', y='perf_per_joule', data=perf_by_agent, ax=ax8)
    ax8.set_title('Performance per Joule (Higher is Better)')
    ax8.set_xlabel('Agent')
    ax8.set_ylabel('Performance/Joule')
    ax8.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_visualizations(rl_agent, env, results_df):
    """
    Generate all visualizations.
    
    Args:
        rl_agent: The trained RL agent
        env: The environment
        results_df: DataFrame with simulation results
    """
    print("Generating visualizations...")
    
    # Plot decision boundaries
    plot_decision_boundaries(rl_agent, env)
    
    # Plot reward landscape
    plot_reward_landscape(env)
    
    # Plot performance over time
    plot_performance_over_time(results_df)
    
    # Plot offloading rate
    plot_offloading_rate(results_df)
    
    # Plot energy efficiency
    plot_energy_efficiency(results_df)
    
    # Plot deadline violations
    plot_deadline_violations(results_df)
    
    # Create comprehensive dashboard
    create_comprehensive_dashboard(results_df, rl_agent, env)
    
    print("Visualizations completed!")

def collect_simulation_results(agents, env, steps=1000):
    """
    Collect detailed results from simulation runs.
    
    Args:
        agents: Dictionary of agents to evaluate
        env: The environment
        steps: Number of steps to simulate
        
    Returns:
        results_df: DataFrame with simulation results
    """
    results = []
    
    for agent_name, agent in agents.items():
        print(f"Collecting data for {agent_name}...")
        state = env.reset()
        
        for step in range(steps):
            # Get task and network conditions
            if env.current_task is None:
                if not env.task_queue:
                    env.current_task = env.generate_random_task()
                else:
                    env.current_task = env.task_queue.popleft()
            
            task = env.current_task
            latency, bandwidth = env.network.get_current_conditions()
            
            # Get normalized state values for recording
            task_complexity = task.complexity / 1e9
            data_size = task.data_size / 5000
            network_latency = latency / 200
            network_bandwidth = bandwidth / 10000
            energy_level = env.edge_device.energy_remaining / env.edge_device.energy_capacity
            
            # Get action based on agent type
            if agent_name == 'RL':
                action = agent.act(state)
            else:
                # Heuristic agent
                action = agent.act(
                    state, env.edge_device, env.cloud_server, 
                    task, (latency, bandwidth)
                )
            
            # Take the action
            next_state, reward, done, info = env.step(action)
            
            # Record result
            results.append({
                'agent': agent_name,
                'step': step,
                'task_id': task.task_id,
                'task_complexity': task_complexity,
                'data_size': data_size,
                'network_latency': network_latency,
                'network_bandwidth': network_bandwidth,
                'energy_level': energy_level,
                'action': action,
                'execution_time': info['execution_time'],
                'energy_consumed': info['energy_consumed'],
                'cloud_cost': info['cost'],
                'deadline_violated': 1 if info['deadline_violated'] else 0,
                'reward': reward
            })
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if done:
                print(f"Episode ended at step {step} due to energy depletion")
                break
    
    # Convert results to DataFrame
    return pd.DataFrame(results)

def main():
    """Main function to run all visualizations."""
    # Assuming we have a trained RL agent, environment, and simulation results
    # This would typically come from running the actual simulation
    
    # Create a placeholder for demonstration
    print("This module provides visualization tools for the Edge Computing Task Offloading project")
    print("Import and use these functions with your trained agents and simulation results")
    
if __name__ == "__main__":
    main()
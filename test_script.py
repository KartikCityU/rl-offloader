"""
Test script for the AI-Driven Task Offloading Algorithm for Edge Computing.
This script demonstrates the functionality of the algorithm and generates
visualizations for analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time
import seaborn as sns  # Add this import for the heatmap visualization

# Import our modules
from edge_offloading_rl import (
    Task, EdgeDevice, CloudServer, NetworkCondition, EdgeEnvironment, 
    DQNAgent, HeuristicAgent, train_rl_agent, evaluate_agent, compare_agents
)

from visualization_tools import (
    plot_decision_boundaries, plot_reward_landscape, plot_performance_over_time,
    plot_offloading_rate, plot_energy_efficiency, plot_deadline_violations,
    create_comprehensive_dashboard, collect_simulation_results
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def main():
    """Main test function."""
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    print("Initializing edge computing environment...")
    
    # Create edge computing environment
    edge_device = EdgeDevice(
        device_id=1,
        cpu_speed=1e9,  # 1 GHz
        energy_capacity=1000,  # 1000 J
        energy_per_cycle=1e-9,  # 1 nJ per cycle
        idle_power=0.1  # 0.1 W
    )
    
    cloud_server = CloudServer(
        server_id=1,
        cpu_speed=5e9,  # 5 GHz
        cost_per_cycle=1e-10  # $0.1 per billion cycles
    )
    
    network = NetworkCondition(
        base_latency=50,  # 50 ms
        latency_variance=30,  # ±30 ms
        base_bandwidth=1000,  # 1000 KB/s
        bandwidth_variance=500  # ±500 KB/s
    )
    
    env = EdgeEnvironment(edge_device, cloud_server, network)
    
    # Display environment configuration
    print("\nEnvironment Configuration:")
    print(f"Edge Device: CPU Speed = {edge_device.cpu_speed/1e6} MHz, Energy Capacity = {edge_device.energy_capacity} J")
    print(f"Cloud Server: CPU Speed = {cloud_server.cpu_speed/1e6} MHz, Cost per Cycle = ${cloud_server.cost_per_cycle * 1e9}/billion cycles")
    print(f"Network: Base Latency = {network.base_latency} ms, Base Bandwidth = {network.base_bandwidth} KB/s")
    
    # Train RL agent with smaller episode count for testing
    print("\nTraining RL agent...")
    rl_agent, training_history = train_rl_agent(env, episodes=100, batch_size=32)
    
    # Create heuristic agents for comparison
    print("\nCreating heuristic agents for comparison...")
    heuristic_agents = {
        'Threshold': HeuristicAgent('threshold'),
        'Energy': HeuristicAgent('energy'),
        'Latency': HeuristicAgent('latency')
    }
    
    # Combine agents for testing
    all_agents = {
        'RL': rl_agent,
        'Threshold': heuristic_agents['Threshold'],
        'Energy': heuristic_agents['Energy'],
        'Latency': heuristic_agents['Latency']
    }
    
    # Collect simulation results
    print("\nRunning simulations and collecting results...")
    results_df = collect_simulation_results(all_agents, env, steps=100)  # Reduced steps for faster execution
    
    # Save results to CSV
    results_df.to_csv('results/simulation_results.csv', index=False)
    print(f"Saved simulation results to 'results/simulation_results.csv'")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot decision boundaries for the RL agent
    print("Generating decision boundaries plot...")
    try:
        plot_decision_boundaries(rl_agent, env)
    except Exception as e:
        print(f"Warning: Could not generate decision boundaries plot: {e}")
    
    # Plot reward landscape
    print("Generating reward landscape plot...")
    try:
        plot_reward_landscape(env)
    except Exception as e:
        print(f"Warning: Could not generate reward landscape plot: {e}")
    
    # Plot performance over time
    print("Generating performance over time plot...")
    try:
        plot_performance_over_time(results_df)
    except Exception as e:
        print(f"Warning: Could not generate performance over time plot: {e}")
    
    # Plot offloading rate by task complexity
    print("Generating offloading rate plot...")
    try:
        plot_offloading_rate(results_df)
    except Exception as e:
        print(f"Warning: Could not generate offloading rate plot: {e}")
    
    # Plot energy efficiency
    print("Generating energy efficiency plot...")
    try:
        plot_energy_efficiency(results_df)
    except Exception as e:
        print(f"Warning: Could not generate energy efficiency plot: {e}")
    
    # Plot deadline violations
    print("Generating deadline violations plot...")
    try:
        plot_deadline_violations(results_df)
    except Exception as e:
        print(f"Warning: Could not generate deadline violations plot: {e}")
    
    # Create comprehensive dashboard
    print("Generating comprehensive dashboard...")
    try:
        create_comprehensive_dashboard(results_df, rl_agent, env)
    except Exception as e:
        print(f"Warning: Could not generate comprehensive dashboard: {e}")
    
    # Compare agent performances
    print("\nComparing agent performances...")
    try:
        comparison_results = compare_agents(env, rl_agent, heuristic_agents, episodes=20)  # Reduced episodes for faster execution
        
        # Print detailed comparison results
        print("\nDetailed comparison results:")
        for agent_name, metrics in comparison_results.items():
            print(f"\n{agent_name} Agent:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Warning: Could not compare agent performances: {e}")
    
    # Test individual tasks
    print("\nTesting individual task scenarios...")
    try:
        test_individual_tasks(env, all_agents)
    except Exception as e:
        print(f"Warning: Could not run individual task tests: {e}")
    
    print("\nTest script completed successfully!")
    print("All results and visualizations have been saved to the 'results' directory.")

def test_individual_tasks(env, agents):
    """Test specific task scenarios with all agents."""
    # Reset environment
    env.reset()
    
    # Define test scenarios
    test_scenarios = [
        {
            'name': 'Light Task, Good Network',
            'task': Task(1, 1e7, 100, 5000),  # Light computation, small data, flexible deadline
            'network': (20, 2000)  # Low latency, high bandwidth
        },
        {
            'name': 'Heavy Task, Good Network',
            'task': Task(2, 5e8, 200, 2000),  # Heavy computation, medium data, tight deadline
            'network': (20, 2000)  # Low latency, high bandwidth
        },
        {
            'name': 'Light Task, Poor Network',
            'task': Task(3, 1e7, 100, 5000),  # Light computation, small data, flexible deadline
            'network': (100, 500)  # High latency, low bandwidth
        },
        {
            'name': 'Heavy Task, Poor Network',
            'task': Task(4, 5e8, 200, 2000),  # Heavy computation, medium data, tight deadline
            'network': (100, 500)  # High latency, low bandwidth
        },
    ]
    
    # Run tests and collect results
    scenario_results = []
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario['name']}")
        task = scenario['task']
        latency, bandwidth = scenario['network']
        
        # Create state representation
        normalized_task_complexity = task.complexity / 1e9
        normalized_data_size = task.data_size / 5000
        normalized_latency = latency / 200
        normalized_bandwidth = bandwidth / 10000
        normalized_energy = env.edge_device.energy_remaining / env.edge_device.energy_capacity
        normalized_deadline = task.deadline / 10000 if task.deadline else 1.0
        
        state = np.array([
            normalized_task_complexity,
            normalized_data_size,
            normalized_latency,
            normalized_bandwidth,
            normalized_energy,
            normalized_deadline
        ])
        
        # Get local and cloud execution estimates for reference
        local_time, local_energy = env.edge_device.get_local_execution_estimate(task)
        cloud_time, cloud_cost = env.cloud_server.get_cloud_execution_estimate(task, latency, bandwidth)
        
        print(f"Task: {task.complexity/1e6:.1f}M cycles, {task.data_size}KB, deadline={task.deadline}ms")
        print(f"Network: Latency={latency}ms, Bandwidth={bandwidth}KB/s")
        print(f"Estimated local execution: {local_time:.2f}ms, {local_energy:.4f}J")
        print(f"Estimated cloud execution: {cloud_time:.2f}ms, ${cloud_cost:.6f}")
        
        # Get decisions from all agents
        for agent_name, agent in agents.items():
            if agent_name == 'RL':
                action = agent.act(state)
            else:
                # Heuristic agent
                action = agent.act(
                    state, env.edge_device, env.cloud_server, 
                    task, (latency, bandwidth)
                )
            
            execution_location = "LOCAL" if action == 0 else "CLOUD"
            expected_time = local_time if action == 0 else cloud_time
            expected_energy = local_energy if action == 0 else 0
            expected_cost = 0 if action == 0 else cloud_cost
            
            print(f"{agent_name} Agent decision: {execution_location}")
            
            # Record result
            scenario_results.append({
                'scenario': scenario['name'],
                'agent': agent_name,
                'decision': execution_location,
                'expected_time': expected_time,
                'expected_energy': expected_energy,
                'expected_cost': expected_cost,
                'decision_numeric': 1 if execution_location == 'CLOUD' else 0  # Add numeric version
            })
    
    # Convert results to DataFrame
    scenario_df = pd.DataFrame(scenario_results)
    
    # Save results
    scenario_df.to_csv('results/scenario_test_results.csv', index=False)
    print(f"\nSaved scenario test results to 'results/scenario_test_results.csv'")
    
    try:
        # Create summary plot
        plt.figure(figsize=(15, 10))
        
        # For the decision heatmap, use the numeric decision field directly
        decision_pivot = pd.pivot_table(
            scenario_df,
            index='scenario',
            columns='agent',
            values='decision_numeric',
            aggfunc='mean'  # This will work because we're using 0 and 1
        )
        
        plt.subplot(2, 1, 1)
        sns.heatmap(decision_pivot, cmap='coolwarm', annot=True, fmt='.0f', cbar_kws={'label': 'Decision (0=Local, 1=Cloud)'})
        plt.title('Task Offloading Decisions by Agent and Scenario')
        
        # Plot expected execution time
        time_pivot = pd.pivot_table(
            scenario_df,
            index='scenario',
            columns='agent',
            values='expected_time',
            aggfunc='mean'
        )
        
        plt.subplot(2, 1, 2)
        sns.heatmap(time_pivot, cmap='viridis', annot=True, fmt='.1f', cbar_kws={'label': 'Expected Execution Time (ms)'})
        plt.title('Expected Execution Time by Agent and Scenario')
        
        plt.tight_layout()
        plt.savefig('results/scenario_test_summary.png', dpi=300, bbox_inches='tight')
        print("Saved scenario summary visualization to 'results/scenario_test_summary.png'")
    except Exception as e:
        print(f"Warning: Could not create scenario summary plot due to: {e}")
    
if __name__ == "__main__":
    main()
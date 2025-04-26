import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class Task:
    """Represents a computational task in the edge computing environment."""
    
    def __init__(self, task_id, complexity, data_size, deadline=None):
        """
        Initialize a new task.
        
        Args:
            task_id: Unique identifier for the task
            complexity: Computational complexity (in CPU cycles)
            data_size: Size of input/output data (in KB)
            deadline: Optional deadline for task completion (in ms)
        """
        self.task_id = task_id
        self.complexity = complexity
        self.data_size = data_size
        self.deadline = deadline
        
    def __str__(self):
        return f"Task {self.task_id}: {self.complexity} cycles, {self.data_size} KB"

class EdgeDevice:
    """Represents an edge computing device with limited resources."""
    
    def __init__(self, device_id, cpu_speed, energy_capacity, energy_per_cycle, idle_power):
        """
        Initialize a new edge device.
        
        Args:
            device_id: Unique identifier for the device
            cpu_speed: Processing speed (in cycles per second)
            energy_capacity: Total energy capacity (in joules)
            energy_per_cycle: Energy consumed per CPU cycle (in joules)
            idle_power: Power consumption when idle (in watts)
        """
        self.device_id = device_id
        self.cpu_speed = cpu_speed
        self.energy_capacity = energy_capacity
        self.energy_remaining = energy_capacity
        self.energy_per_cycle = energy_per_cycle
        self.idle_power = idle_power
        self.busy = False
        
    def execute_task(self, task):
        """
        Execute a task locally on the edge device.
        
        Args:
            task: The task to execute
            
        Returns:
            execution_time: Time taken to execute (in ms)
            energy_consumed: Energy consumed during execution (in joules)
        """
        execution_time = (task.complexity / self.cpu_speed) * 1000  # Convert to ms
        energy_consumed = task.complexity * self.energy_per_cycle
        
        # Update energy remaining
        self.energy_remaining = max(0, self.energy_remaining - energy_consumed)
        
        return execution_time, energy_consumed
    
    def get_local_execution_estimate(self, task):
        """
        Estimate time and energy for local execution without actually executing.
        
        Args:
            task: The task to estimate
            
        Returns:
            execution_time: Estimated time to execute (in ms)
            energy_consumed: Estimated energy to consume (in joules)
        """
        execution_time = (task.complexity / self.cpu_speed) * 1000  # Convert to ms
        energy_consumed = task.complexity * self.energy_per_cycle
        
        return execution_time, energy_consumed
    
    def get_energy_percentage(self):
        """Get the percentage of energy remaining."""
        return (self.energy_remaining / self.energy_capacity) * 100

class CloudServer:
    """Represents a cloud server with high computational resources."""
    
    def __init__(self, server_id, cpu_speed, cost_per_cycle):
        """
        Initialize a new cloud server.
        
        Args:
            server_id: Unique identifier for the server
            cpu_speed: Processing speed (in cycles per second)
            cost_per_cycle: Cost per CPU cycle (in $)
        """
        self.server_id = server_id
        self.cpu_speed = cpu_speed
        self.cost_per_cycle = cost_per_cycle
        
    def execute_task(self, task, network_latency, bandwidth):
        """
        Execute a task on the cloud server.
        
        Args:
            task: The task to execute
            network_latency: Round-trip network latency (in ms)
            bandwidth: Network bandwidth (in KB/s)
            
        Returns:
            execution_time: Time taken to execute including network transfer (in ms)
            cost: Financial cost of execution (in $)
        """
        # Calculate transfer time
        transfer_time = (task.data_size / bandwidth) * 1000  # Convert to ms
        
        # Calculate processing time
        processing_time = (task.complexity / self.cpu_speed) * 1000  # Convert to ms
        
        # Total execution time including network latency
        execution_time = network_latency + transfer_time + processing_time
        
        # Calculate cost
        cost = task.complexity * self.cost_per_cycle
        
        return execution_time, cost
    
    def get_cloud_execution_estimate(self, task, network_latency, bandwidth):
        """
        Estimate time and cost for cloud execution without actually executing.
        
        Args:
            task: The task to estimate
            network_latency: Round-trip network latency (in ms)
            bandwidth: Network bandwidth (in KB/s)
            
        Returns:
            execution_time: Estimated time to execute (in ms)
            cost: Estimated financial cost (in $)
        """
        # Calculate transfer time
        transfer_time = (task.data_size / bandwidth) * 1000  # Convert to ms
        
        # Calculate processing time
        processing_time = (task.complexity / self.cpu_speed) * 1000  # Convert to ms
        
        # Total execution time including network latency
        execution_time = network_latency + transfer_time + processing_time
        
        # Calculate cost
        cost = task.complexity * self.cost_per_cycle
        
        return execution_time, cost

class NetworkCondition:
    """Represents the network conditions between edge device and cloud."""
    
    def __init__(self, base_latency, latency_variance, base_bandwidth, bandwidth_variance):
        """
        Initialize network conditions.
        
        Args:
            base_latency: Base network latency (in ms)
            latency_variance: Maximum variance in latency (in ms)
            base_bandwidth: Base network bandwidth (in KB/s)
            bandwidth_variance: Maximum variance in bandwidth (in KB/s)
        """
        self.base_latency = base_latency
        self.latency_variance = latency_variance
        self.base_bandwidth = base_bandwidth
        self.bandwidth_variance = bandwidth_variance
        
    def get_current_conditions(self):
        """
        Get current network conditions with random variance.
        
        Returns:
            current_latency: Current network latency (in ms)
            current_bandwidth: Current network bandwidth (in KB/s)
        """
        latency_offset = np.random.uniform(-self.latency_variance, self.latency_variance)
        current_latency = max(1, self.base_latency + latency_offset)
        
        bandwidth_offset = np.random.uniform(-self.bandwidth_variance, self.bandwidth_variance)
        current_bandwidth = max(1, self.base_bandwidth + bandwidth_offset)
        
        return current_latency, current_bandwidth

class EdgeEnvironment:
    """Simulates the edge computing environment for reinforcement learning."""
    
    def __init__(self, edge_device, cloud_server, network):
        """
        Initialize the edge computing environment.
        
        Args:
            edge_device: The edge device
            cloud_server: The cloud server
            network: The network conditions
        """
        self.edge_device = edge_device
        self.cloud_server = cloud_server
        self.network = network
        self.current_task = None
        self.task_queue = deque()
        self.completed_tasks = []
        self.total_energy_consumed = 0
        self.total_cloud_cost = 0
        self.total_execution_time = 0
        self.time_violations = 0
        
    def add_task(self, task):
        """Add a task to the queue."""
        self.task_queue.append(task)
        
    def generate_random_task(self):
        """Generate a random task with realistic parameters."""
        task_id = len(self.completed_tasks) + len(self.task_queue) + 1
        complexity = np.random.randint(1e6, 1e9)  # Between 1M and 1B cycles
        data_size = np.random.randint(10, 5000)  # Between 10KB and 5MB
        deadline = np.random.randint(100, 10000)  # Between 100ms and 10s
        
        return Task(task_id, complexity, data_size, deadline)
    
    def get_state(self):
        """
        Get the current state of the environment for RL agent.
        
        Returns:
            state: Array representing the current state
        """
        if not self.current_task:
            if not self.task_queue:
                self.current_task = self.generate_random_task()
            else:
                self.current_task = self.task_queue.popleft()
                
        # Get current network conditions
        latency, bandwidth = self.network.get_current_conditions()
        
        # Normalize state values to [0, 1] range for better RL performance
        normalized_task_complexity = self.current_task.complexity / 1e9
        normalized_data_size = self.current_task.data_size / 5000
        normalized_latency = latency / 200
        normalized_bandwidth = bandwidth / 10000
        normalized_energy = self.edge_device.energy_remaining / self.edge_device.energy_capacity
        
        if self.current_task.deadline:
            normalized_deadline = self.current_task.deadline / 10000
            state = np.array([
                normalized_task_complexity,
                normalized_data_size,
                normalized_latency,
                normalized_bandwidth,
                normalized_energy,
                normalized_deadline
            ])
        else:
            state = np.array([
                normalized_task_complexity,
                normalized_data_size,
                normalized_latency,
                normalized_bandwidth,
                normalized_energy,
                1.0  # Default normalized deadline when none exists
            ])
            
        return state
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: 0 for local execution, 1 for cloud offloading
            
        Returns:
            next_state: The next state
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        if not self.current_task:
            if not self.task_queue:
                self.current_task = self.generate_random_task()
            else:
                self.current_task = self.task_queue.popleft()
        
        # Get current network conditions
        latency, bandwidth = self.network.get_current_conditions()
        
        # Execute the task based on the action
        if action == 0:  # Local execution
            execution_time, energy_consumed = self.edge_device.execute_task(self.current_task)
            cost = 0
            self.total_energy_consumed += energy_consumed
        else:  # Cloud offloading
            execution_time, cost = self.cloud_server.execute_task(
                self.current_task, latency, bandwidth
            )
            energy_consumed = 0  # Simplified assumption
            self.total_cloud_cost += cost
        
        # Check deadline violation if deadline exists
        deadline_violated = False
        if self.current_task.deadline and execution_time > self.current_task.deadline:
            deadline_violated = True
            self.time_violations += 1
        
        # Calculate reward
        reward = self.calculate_reward(execution_time, energy_consumed, cost, deadline_violated)
        
        # Update total execution time
        self.total_execution_time += execution_time
        
        # Add task to completed tasks
        self.completed_tasks.append((self.current_task, action, execution_time, energy_consumed, cost))
        
        # Clear current task
        self.current_task = None
        
        # Check if episode is done (e.g., battery depleted)
        done = self.edge_device.energy_remaining <= 0
        
        # Get next state
        next_state = self.get_state()
        
        # Additional info
        info = {
            'execution_time': execution_time,
            'energy_consumed': energy_consumed,
            'cost': cost,
            'deadline_violated': deadline_violated
        }
        
        return next_state, reward, done, info
    
    def calculate_reward(self, execution_time, energy_consumed, cost, deadline_violated):
        """
        Calculate the reward for the current action.
        
        Args:
            execution_time: Time taken for execution (in ms)
            energy_consumed: Energy consumed (in joules)
            cost: Financial cost (in $)
            deadline_violated: Whether deadline was violated
            
        Returns:
            reward: The calculated reward
        """
        # Normalize factors
        norm_time = min(1.0, 10000 / max(1, execution_time))  # Higher for faster execution
        norm_energy = 1.0 - min(1.0, energy_consumed / (self.edge_device.energy_capacity * 0.1))  # Higher for less energy
        norm_cost = 1.0 - min(1.0, cost / 0.01)  # Higher for lower cost, assuming 0.01$ is high
        
        # Weights for different factors (can be tuned)
        w_time = 0.4
        w_energy = 0.3
        w_cost = 0.3
        
        # Base reward is weighted sum of normalized factors
        reward = w_time * norm_time + w_energy * norm_energy + w_cost * norm_cost
        
        # Penalty for deadline violation
        if deadline_violated:
            reward *= 0.5
        
        return reward
    
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            state: Initial state after reset
        """
        self.edge_device.energy_remaining = self.edge_device.energy_capacity
        self.current_task = None
        self.completed_tasks = []
        self.total_energy_consumed = 0
        self.total_cloud_cost = 0
        self.total_execution_time = 0
        self.time_violations = 0
        
        return self.get_state()

class DQNAgent:
    """Deep Q-Network agent for task offloading decisions."""
    
    def __init__(self, state_size, action_size):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build a neural network model for DQN."""
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Update the target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model using experiences from memory."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(np.array([next_state]), verbose=0)[0]
                )
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
    
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)

class HeuristicAgent:
    """Baseline heuristic agent for task offloading decisions."""
    
    def __init__(self, heuristic_type='threshold'):
        """
        Initialize the heuristic agent.
        
        Args:
            heuristic_type: Type of heuristic to use ('threshold', 'energy', 'latency')
        """
        self.heuristic_type = heuristic_type
    
    def act(self, state, edge_device, cloud_server, task, network_conditions):
        """
        Choose an action based on the heuristic.
        
        Args:
            state: Current state (unused in heuristics)
            edge_device: The edge device
            cloud_server: The cloud server
            task: The current task
            network_conditions: Current network conditions
            
        Returns:
            action: 0 for local execution, 1 for cloud offloading
        """
        latency, bandwidth = network_conditions
        
        if self.heuristic_type == 'threshold':
            # Threshold-based heuristic
            # Offload if task complexity is high or energy is low
            complexity_threshold = 5e8  # 500M cycles
            energy_threshold = 30  # 30% remaining energy
            
            if (task.complexity > complexity_threshold or 
                edge_device.get_energy_percentage() < energy_threshold):
                return 1  # Offload to cloud
            else:
                return 0  # Execute locally
                
        elif self.heuristic_type == 'energy':
            # Energy-efficient heuristic
            # Compare energy consumption for local execution
            local_time, local_energy = edge_device.get_local_execution_estimate(task)
            
            # If energy is low or task would consume too much energy, offload
            energy_threshold = 20  # 20% remaining energy
            if (edge_device.get_energy_percentage() < energy_threshold or 
                local_energy > edge_device.energy_remaining * 0.5):
                return 1  # Offload to cloud
            else:
                return 0  # Execute locally
                
        elif self.heuristic_type == 'latency':
            # Latency-aware heuristic
            # Compare execution times
            local_time, _ = edge_device.get_local_execution_estimate(task)
            cloud_time, _ = cloud_server.get_cloud_execution_estimate(task, latency, bandwidth)
            
            if cloud_time < local_time:
                return 1  # Offload to cloud
            else:
                return 0  # Execute locally
                
        else:
            # Default random choice
            return random.randint(0, 1)

def evaluate_agent(agent, env, episodes=100, agent_type='RL'):
    """
    Evaluate an agent's performance.
    
    Args:
        agent: The agent to evaluate
        env: The environment
        episodes: Number of episodes to evaluate
        agent_type: Type of agent ('RL' or 'Heuristic')
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    total_rewards = []
    execution_times = []
    energy_consumptions = []
    cloud_costs = []
    deadline_violations = []
    
    for e in range(episodes):
        state = env.reset()
        cumulative_reward = 0
        done = False
        step_count = 0
        max_steps = 100  # Maximum steps per episode
        
        while not done and step_count < max_steps:
            if agent_type == 'RL':
                action = agent.act(state)
            else:  # Heuristic
                if env.current_task is None:
                    if not env.task_queue:
                        env.current_task = env.generate_random_task()
                    else:
                        env.current_task = env.task_queue.popleft()
                        
                latency, bandwidth = env.network.get_current_conditions()
                action = agent.act(
                    state, env.edge_device, env.cloud_server, 
                    env.current_task, (latency, bandwidth)
                )
            
            next_state, reward, done, info = env.step(action)
            cumulative_reward += reward
            
            execution_times.append(info['execution_time'])
            energy_consumptions.append(info['energy_consumed'])
            cloud_costs.append(info['cost'])
            deadline_violations.append(1 if info['deadline_violated'] else 0)
            
            state = next_state
            step_count += 1
        
        total_rewards.append(cumulative_reward)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(total_rewards),
        'mean_execution_time': np.mean(execution_times),
        'mean_energy_consumption': np.mean(energy_consumptions),
        'mean_cloud_cost': np.mean(cloud_costs),
        'deadline_violation_rate': np.mean(deadline_violations) * 100
    }
    
    return metrics

def train_rl_agent(env, episodes=1000, batch_size=32):
    """
    Train the RL agent.
    
    Args:
        env: The environment
        episodes: Number of episodes to train
        batch_size: Batch size for replay
        
    Returns:
        agent: The trained agent
        training_history: Training metrics history
    """
    state_size = env.get_state().shape[0]
    action_size = 2  # Local or offload
    agent = DQNAgent(state_size, action_size)
    
    training_history = {
        'rewards': [],
        'execution_times': [],
        'energy_consumptions': [],
        'cloud_costs': [],
        'epsilon': []
    }
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        max_steps = 100  # Maximum steps per episode
        
        while not done and step_count < max_steps:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Track metrics for this step
            training_history['execution_times'].append(info['execution_time'])
            training_history['energy_consumptions'].append(info['energy_consumed'])
            training_history['cloud_costs'].append(info['cost'])
            
            step_count += 1
        
        # Train the agent using experience replay
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Update target model periodically
        if e % 10 == 0:
            agent.update_target_model()
        
        # Track episode metrics
        training_history['rewards'].append(total_reward)
        training_history['epsilon'].append(agent.epsilon)
        
        # Print progress
        if (e + 1) % 100 == 0:
            print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    return agent, training_history

def compare_agents(env, rl_agent, heuristic_agents, episodes=100):
    """
    Compare performance of different agents.
    
    Args:
        env: The environment
        rl_agent: The trained RL agent
        heuristic_agents: Dictionary of heuristic agents
        episodes: Number of episodes to evaluate
        
    Returns:
        results: Comparison results
    """
    results = {}
    
    # Evaluate RL agent
    print("Evaluating RL agent...")
    rl_metrics = evaluate_agent(rl_agent, env, episodes, 'RL')
    results['RL'] = rl_metrics
    
    # Evaluate heuristic agents
    for name, agent in heuristic_agents.items():
        print(f"Evaluating {name} heuristic...")
        heuristic_metrics = evaluate_agent(agent, env, episodes, 'Heuristic')
        results[name] = heuristic_metrics
    
    return results

def plot_comparison(results, metrics_to_plot):
    """
    Plot comparison of agent performances.
    
    Args:
        results: Comparison results
        metrics_to_plot: List of metrics to plot
    """
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics_to_plot):
        metric_values = []
        labels = []
        
        for agent_name, metrics in results.items():
            if metric in metrics:
                metric_values.append(metrics[metric])
                labels.append(agent_name)
        
        axes[i].bar(labels, metric_values)
        axes[i].set_title(f"{metric}")
        axes[i].set_ylabel(metric)
        
        # Add values on top of bars
        for j, value in enumerate(metric_values):
            axes[i].text(j, value, f"{value:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('agent_comparison.png')
    plt.close()

def plot_training_history(history):
    """
    Plot training history of the RL agent.
    
    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(history['rewards'])
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Plot execution times
    axes[0, 1].plot(history['execution_times'])
    axes[0, 1].set_title('Execution Times')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Time (ms)')
    
    # Plot energy consumption
    axes[1, 0].plot(history['energy_consumptions'])
    axes[1, 0].set_title('Energy Consumption')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Energy (J)')
    
    # Plot cloud costs
    axes[1, 1].plot(history['cloud_costs'])
    axes[1, 1].set_title('Cloud Costs')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Cost ($)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
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
    
    # Train RL agent
    print("Training RL agent...")
    rl_agent, training_history = train_rl_agent(env, episodes=500, batch_size=32)
    
    # Save trained model
    rl_agent.save("edge_offloading_model.h5")
    
    # Create heuristic agents for comparison
    heuristic_agents = {
        'Threshold': HeuristicAgent('threshold'),
        'Energy': HeuristicAgent('energy'),
        'Latency': HeuristicAgent('latency')
    }
    
    # Compare agents
    print("Comparing agent performances...")
    comparison_results = compare_agents(env, rl_agent, heuristic_agents, episodes=100)
    
    # Plot results
    metrics_to_plot = [
        'mean_reward', 
        'mean_execution_time', 
        'mean_energy_consumption', 
        'mean_cloud_cost',
        'deadline_violation_rate'
    ]
    plot_comparison(comparison_results, metrics_to_plot)
    
    # Plot training history
    plot_training_history(training_history)
    
    # Print detailed results
    print("\nDetailed comparison results:")
    for agent_name, metrics in comparison_results.items():
        print(f"\n{agent_name} Agent:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
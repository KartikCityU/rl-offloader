# Dynamic AI-Driven Task Offloading in Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of a dynamic AI-driven task offloading algorithm for edge computing environments, as described in our CS4296 Cloud Computing (Spring 2025) research project. The algorithm uses reinforcement learning (RL) to adaptively make decisions about whether to execute tasks locally on edge devices or offload them to cloud servers.

## Authors
- SHARMA Kartik (56584014)
- Dylan Kenneth Winoto (57187530)

## Overview

Edge computing is a paradigm that aims to reduce latency and improve efficiency by processing data closer to its source. However, deciding whether to execute tasks locally on edge devices or offload them to cloud servers is challenging due to factors such as network conditions, device energy levels, task complexity, and deadlines.

Our approach uses Deep Q-Networks (DQN) to learn optimal task offloading policies that adapt to real-time changes in the environment. The system optimizes multiple objectives simultaneously:
- Execution time
- Energy consumption
- Monetary cost
- Deadline satisfaction

## Key Features

- **Reinforcement Learning Framework**: Uses DQN to learn optimal offloading policies
- **Multi-objective Optimization**: Balances execution time, energy consumption, and cost
- **Dynamic Adaptation**: Adapts to changing network conditions and device status
- **Comprehensive Evaluation**: Compare against multiple heuristic-based approaches

## System Architecture

The system consists of:
1. **Edge Devices**: Resource-constrained devices with limited computational capabilities
2. **Cloud Servers**: Remote computing resources with high computational power
3. **Tasks**: Computational workloads with varying complexity, data size, and deadlines
4. **Network**: Communication channel with varying latency and bandwidth

## Repository Structure

```
.
├── edge_offloading_rl.py       # Core implementation of RL-based offloading
├── visualization_tools.py       # Tools for generating visualizations
├── test_script.py               # Scripts for running simulations
├── requirements.txt             # Project dependencies
├── results/                     # Directory containing simulation results
│   ├── decision_boundaries.png
│   ├── performance_over_time.png
│   ├── reward_landscape.png
│   └── comprehensive_dashboard.png
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/rl-offloader.git
cd rl-offloader
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows:
   ```bash
   venv\Scripts\activate
   ```
   - On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The implementation requires the following libraries:
- numpy>=1.19.5
- pandas>=1.3.0
- matplotlib>=3.4.2
- tensorflow>=2.5.0
- seaborn>=0.11.1
- scikit-learn>=0.24.2

## Usage

### Running the Simulation

To run the full simulation with default parameters:

```bash
python test_script.py
```

This will train the RL agent, evaluate it against the baseline heuristics, and generate visualizations in the `results` directory.

### Customizing Experiments

To modify experimental parameters, edit the following in `test_script.py`:

```python
# Configure environment parameters
env_config = {
    'edge_device': {
        'cpu_speed': 1e9,        # 1 GHz
        'energy_capacity': 1000, # 1000 J
        'energy_per_cycle': 1e-9 # 1 nJ/cycle
    },
    'cloud_server': {
        'cpu_speed': 5e9,         # 5 GHz
        'cost_per_cycle': 1e-10   # $0.1 per billion cycles
    },
    'network': {
        'base_latency': 50,       # 50 ms
        'base_bandwidth': 1000    # 1000 KB/s
    }
}

# Run simulation
train_rl_agent(env_config, episodes=1000, batch_size=32)
```

### Analyzing Specific Scenarios

To evaluate specific task scenarios:

```python
# Define a specific task
task = Task(
    complexity=1e6,    # 1M cycles
    data_size=100,     # 100 KB
    deadline=1000      # 1000 ms
)

# Test with different agents
test_individual_task(task, agents=['RL', 'Threshold', 'Energy', 'Latency'])
```

## Results

Our reinforcement learning approach demonstrates significant improvements over traditional heuristic-based methods:

- 23% lower execution time compared to the Threshold heuristic
- 31% lower energy consumption compared to the Latency-aware heuristic
- 15% lower cloud cost compared to the Energy-aware heuristic
- 42% lower deadline violation rate compared to all heuristics

Key visualizations include:
- Decision boundaries across different task complexities and network conditions
- Performance metrics over time
- Energy efficiency analysis
- Deadline violation rates
- Offloading rates by task complexity

## Algorithm Details

The core of our approach is a Deep Q-Network (DQN) that learns the optimal action-value function Q(s,a), which estimates the expected long-term reward of taking action a in state s.

Key components include:
- **State Space**: Task characteristics, network conditions, and device status
- **Action Space**: Binary decision (local execution or cloud offloading)
- **Reward Function**: Multi-objective function combining execution time, energy consumption, and cost
- **Neural Network**: Two fully-connected hidden layers with 24 neurons each

## Future Work

Future research directions include:
1. Multi-device collaboration for collaborative offloading
2. Partial offloading capabilities
3. Improved transfer learning for new environments
4. Hardware implementation validation
5. Federated learning for privacy-preserving offloading decisions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@inproceedings{sharma2025dynamic,
  title={Dynamic AI-Driven Task Offloading in Edge Computing},
  author={Sharma, Kartik and Winoto, Dylan Kenneth},
  booktitle={CS4296 Cloud Computing},
  year={2025},
  organization={City University of Hong Kong}
}
```

## Acknowledgements

We would like to thank our instructor and classmates in the CS4296 Cloud Computing course for their valuable feedback and suggestions.

# 🧠 Reinforcement Learning for Classic Control Environments

This repository provides an implementation of reinforcement learning algorithms—**Dueling DQN** and **Monte Carlo REINFORCE**—to solve classic control environments of **CartPole** and **Acrobot** using PyTorch. Apart from this the Dueling DQN has support for versions where the unidentifiability problem is solved using the mean and the max of the advantage function and the REINFORCE has support for both the vanilla version as well the version with baseline

## 🚀 Features

- Support for **Dueling Deep Q-Network (Dueling DQN)**
- Support for **Monte Carlo REINFORCE**
- Replay buffer for experience replay
- Target network updates
- Customizable training hyperparameters

## File description

| 📄 **File Name**         | 📝 **Description** |
|-------------------------|-------------------|
| `main.py`               | 🚦 Main entry point. Handles argument parsing, configuration, environment setup, and launches training for the selected RL algorithm (Dueling DQN or Monte Carlo REINFORCE). Integrates with Weights & Biases for experiment tracking. |
| `agent.py`              | 🤖 Implements the agent classes: `DuelingDQNAgent` and `MonteCarloREINFORCEAgent`. Handles action selection, learning updates, and (for DQN) experience replay. |
| `algorithms.py`         | 🧩 Contains neural network architectures: `DuelingDQN`, `MonteCarloREINFORCE`, and `ValueNetwork`. These define the models used by the agents. |
| `argument_parser.py`    | 🗝️ Defines and parses all command-line arguments for configuring the environment, algorithm, and hyperparameters. |
| `configure.py`          | 🛠️ Provides the `Configure` class, which maps configuration scripts to the correct environment, agent, algorithm, policy, and training function. |
| `policy.py`             | 🎲 Contains the `Policy` base class and the `EpsGreedy` policy for epsilon-greedy action selection in DQN. |
| `replay_buffer.py`      | 🗃️ Implements the `ReplayBuffer` class for storing and sampling experience tuples, enabling experience replay in DQN. |
| `shared.py`             | 🔗 Stores shared resources such as the device configuration (CPU/GPU) and the global configuration script. |
| `trainer.py`            | 🏋️ Implements the `Trainer` class, which provides training loops for both Dueling DQN and Monte Carlo REINFORCE algorithms, including logging and early stopping. |

## 📦 Dependencies

- Python 3.7+
- PyTorch
- NumPy
- OpenAI Gym
- argparse


Install dependencies:

## Reproducing the code 
```bash
git clone https://github.com/rudra-iitm-phd/rl_assignment2.git
cd rl_assignment2
```
### 🧾 Command Line Arguments


| 🏷️ **Argument**       | ✂️ **Short Name** | 🔢 **Type**   | ⚙️ **Default Value**  | 📝 **Description**                                                                 |
|------------------------|-------------------|---------------|------------------------|------------------------------------------------------------------------------------|
| `--env`               | `-env`           | `str`         | `'cartpole'`          | 🌍 Choices of environment: `acrobot`.                                             |
| `--algorithm`         | `-a`             | `str`         | `'dueling_dqn'`       | 🧠 Choices of algorithm: `dueling_dqn`, `mc_reinforce`.                           |
| `--use_max`           | `-um`            | `bool`        | `False`               | 📈 Use the max to constrain the advantage function.                               |
| `--buffer_size`       | `-bs`            | `int`         | `1e5`                 | 🗂️ Size of the replay buffer.                                                     |
| `--learning_rate`     | `-lr`            | `float`       | `1e-4`                | ⚡ Learning rate for the optimizer.                                               |
| `--value_learning_rate` | `-vlr`         | `float`       | `1e-3`                | 📊 Learning rate for the value optimizer.                                         |
| `--batch_size`        | `-b`             | `int`         | `128`                 | 📦 Batch size.                                                                    |
| `--gamma`             | `-g`             | `float`       | `0.99`                | 🔄 Gamma value used for discounting.                                              |
| `--update_period`     | `-up`            | `int`         | `4`                   | ⏳ Time period for updating the target network parameters.                        |
| `--use_baseline`      | `-ub`            | `bool`        | `False`               | 🧮 Use baseline for Monte Carlo REINFORCE.                                        |


### 🧪 Example

Train the agent to solve the acrobot environment using monte carlo reinforce with the baseline version

```bash
python main.py -env acrobot -a mc_reinforce -lr 1e-3 -ub True
```

Train the agent to solve the cartpole environment using Dueling DQN using the max version for solving the unidentifiability problem

```bash
python main.py -env cartpole -a dueling_dqn -um True
```
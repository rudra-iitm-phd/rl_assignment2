# 🧠 Reinforcement Learning for Classic Control Environments

This repository provides an implementation of reinforcement learning algorithms—**Dueling DQN** and **Monte Carlo REINFORCE**—to solve classic control environments such as **CartPole** and **Acrobot** using PyTorch.

## 🚀 Features

- Support for **Dueling Deep Q-Network (Dueling DQN)**
- Support for **Monte Carlo REINFORCE**
- Replay buffer for experience replay
- Target network updates
- Customizable training hyperparameters

## 📦 Dependencies

- Python 3.7+
- PyTorch
- NumPy
- OpenAI Gym
- argparse

Install dependencies:


### 🧾 Command Line Arguments

| Argument           | Short | Type   | Default     | Description                                                  |
|--------------------|-------|--------|-------------|--------------------------------------------------------------|
| `--env`            | `-env`| `str`  | `'cartpole'`| Environment to train on. Example: `'cartpole'`, `'acrobot'`  |
| `--algorithm`      | `-a`  | `str`  | `'dueling_dqn'` | Algorithm to use: `'dueling_dqn'` or `'mc_reinforce'`     |
| `--use_max`        | `-um` | `bool` | `False`     | Whether to constrain the advantage function using max        |
| `--buffer_size`    | `-bs` | `int`  | `100000`    | Size of the experience replay buffer                         |
| `--learning_rate`  | `-lr` | `float`| `1e-4`      | Learning rate for optimizer                                  |
| `--batch_size`     | `-b`  | `int`  | `128`       | Batch size for training                                      |
| `--gamma`          | `-g`  | `float`| `0.99`      | Discount factor for future rewards                           |
| `--update_period`  | `-up` | `int`  | `4`         | How often to update the target network                       |

### 🧪 Example

```bash
python main.py -um True -env acrobot
```
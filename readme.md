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

```bash
python main.py -env acrobot -a mc_reinforce -lr 1e-3 -ub True
```
import argparse

parser = argparse.ArgumentParser(description = "Train a reinforecement learning agent to solve classic control environments using Dueling DQN and Monte Carlo REINFORCE")

parser.add_argument('-env', '--env',
                    type = str, default = 'cartpole',
                    help = "Choices of environment : acrobot")

parser.add_argument('-a', '--algorithm',
                    type = str, default = 'dueling_dqn',
                    help = "Choices of environment : dueling_dqn, mc_reinforce")

parser.add_argument('-um', '--use_max',
                    type = bool, default = False,
                    help = "Use the max to constrain the advantage function")

parser.add_argument('-bs', '--buffer_size',
                    type = int, default = int(1e5),
                    help = "Size of the replay buffer")

parser.add_argument('-lr', '--learning_rate',
                    type = float, default = 1e-4,
                    help = "Learning rate for the optimizer")

parser.add_argument('-vlr', '--value_learning_rate',
                    type = float, default = 1e-3,
                    help = "Learning rate for the optimizer")

parser.add_argument('-b', '--batch_size',
                    type = int, default = 128,
                    help = "Batch size")

parser.add_argument('-g', '--gamma',
                    type = float, default = 0.99,
                    help = "Gamma value used for discounting")

parser.add_argument('-up', '--update_period',
                    type = int, default = 4,
                    help = "Time period for updating the target network parameters")

parser.add_argument('-ub', '--use_baseline',
                    type = bool, default = False,
                    help = "Use baseline for the Monte Carlo : REINFORCE")

parser.add_argument('--wandb_sweep', action='store_true', help='Enable W&B sweep')

parser.add_argument('--sweep_id', type = str, help = "Sweep ID", default = None)

parser.add_argument('-we', '--wandb_entity', 
                  type = str, default = 'da24d008-iit-madras' ,
                  help = 'Wandb Entity used to track experiments in the Weights & Biases dashboard')

parser.add_argument('-wp', '--wandb_project', 
                  type = str, default = 'rl-pa2' ,
                  help = 'Project name used to track experiments in Weights & Biases dashboard')
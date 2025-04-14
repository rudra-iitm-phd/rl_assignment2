sweep_config = {
      "method": "bayes",
            "metric": {"name": "Average return", "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 0.01},
                "algorithm": {"values": ["mc_reinforce", "dueling_dqn"]},
                "use_baseline":{"values":[True, False]},
                "batch_size":{"values":[64, 128, 256]},
                "env":{"values":["cartpole"]},
                "buffer_size":{"values":[100, 1000, 10000, 10000]},
                "update_period":{"values":[2, 4, 8, 16, 32, 64]}

            }
}
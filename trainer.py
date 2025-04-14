from collections import deque
import numpy as np
import shared

class Trainer:
    def __init__(self, algorithm:str):
        self.algorithm = algorithm
        self.mapping = {
            "dueling_dqn":self.dueling_trainer,
            "mc_reinforce":self.mc_trainer
        }
        self.t = 0

    def get_trainer(self):
        return self.mapping[self.algorithm]

    def dueling_trainer(self, env, agent, logger, n_episodes=10000, max_t=1000, eps_start=1, eps_end=0.01, eps_decay=0.985):

        scores_window = deque(maxlen=100)
        eps = eps_start

        score_history = []

        for i_episode in range(1, n_episodes+1):
            state, _ = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ , _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)

            eps = max(eps_end, eps_decay*eps)
            ''' decrease epsilon '''

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            logger.log({"Episodic return":score})
            logger.log({"Average return": np.mean(scores_window)})

            self.verbosity(i_episode, np.mean(scores_window))
            if np.mean(scores_window)>=env.spec.reward_threshold:
                self.verbosity(i_episode, np.mean(scores_window), solved = True)
                break
        return True, score_history, i_episode


    def mc_trainer(self, env, agent, logger, n_episodes=5000, max_t=1000):
        scores = []
        for i_episode in range(1, n_episodes+1):
            state, _ = env.reset()
            episode_rewards = []
            
            for t in range(max_t):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.rewards.append(reward)
                if shared.configuration_script['use_baseline']:
                    state_value = agent.get_value(state)
                    agent.values.append(state_value)
                    state = next_state
                episode_rewards.append(reward)
                if done:
                    break
                    

            agent.update_policy()
            
            total_reward = sum(episode_rewards)
            scores.append(total_reward)
            avg_score = np.mean(scores[-100:])

            logger.log({"Average return": avg_score})
            logger.log({"Episodic return":total_reward})
            
            self.verbosity(i_episode, avg_score)

            if avg_score >= env.spec.reward_threshold:  
                self.verbosity(i_episode, avg_score, solved=True )
                break
        return scores                                                       
    
    def verbosity(self, episode, avg_return, update_period=100, solved=False):
        print('\rEpisode {}\tAverage Return: {:.2f}'.format(episode, avg_return), end="")
        if solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, avg_return))   
        elif episode % update_period == 0:
            print('\rEpisode {}\tAverage Return: {:.2f}'.format(episode, avg_return))
                
        

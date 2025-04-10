from collections import deque
import numpy as np

def dueling_trainer(env, agent, n_episodes=10000, max_t=1000, eps_start=1, eps_end=0.01, eps_decay=0.985):

    scores_window = deque(maxlen=100)
    ''' last 100 scores for checking if the avg is more than 195 '''

    eps = eps_start
    ''' initialize epsilon '''

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

        if i_episode % 100 == 0:
           print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
           score_history.append([i_episode, np.mean(scores_window)])
        if np.mean(scores_window)>=env.spec.reward_threshold:
           print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
           score_history.append([i_episode, np.mean(scores_window)])
           break
    return True, score_history, i_episode


def mc_trainer(env, agent, n_episodes=5000, max_t=1000):
    scores = []
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        episode_rewards = []
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            episode_rewards.append(reward)
            if done:
                break
                
        # Update after each episode
        agent.update_policy()
        
        total_reward = sum(episode_rewards)
        scores.append(total_reward)
        
        if i_episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}')
            if avg_score >= env.spec.reward_threshold:  # CartPole solving condition
                print(f"Solved in {i_episode} episodes!")
                break
    return scores


from math import radians
from random import random
import gym


# Code from
# https://gist.github.com/n1try/87b442fce7f7d58606f462191c6d6033
buckets = (1, 1, 6, 12,)   # down-scaling feature space to discrete range
def discretize(obs, env):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, int(round((buckets[i] - 1) * ratios[i])))) for i in range(len(obs))]

    return tuple(new_obs)


def add_state(q, state, action_space):
    if state not in q:
        q.update({state: {x: 0 for x in action_space}})


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    action_space = [i for i in range(env.action_space.n)]

    win_score = 195
    
    n_episodes = 10000
    max_turns = 200

    get_alpha = lambda t: .6 if t < 200 else .1
    get_epsilon = lambda t: 1 if t < 200 else .1
    gamma = 1

    q = {}
    wins = 0
    for episode in range(n_episodes):
        r = 0  # cumulative reward

        alpha = get_alpha(episode)
        epsilon = get_epsilon(episode)

        observation = env.reset()
        state = discretize(observation, env)
        add_state(q, state, action_space)

        for _ in range(max_turns):
            action = max(q[state], key=q[state].get) if random() > epsilon else env.action_space.sample()

            observation, reward, done, info = env.step(action)

            next_state = discretize(observation, env)
            add_state(q, next_state, action_space)

            q[state][action] += alpha * (reward + gamma * max(q[next_state].values()) - q[state][action])

            state = next_state
            
            r += reward

            if done:
                break

        if not episode % 10:
            print(f'Episode: {episode}, Reward: {r}\n\n')
        
        if r >= win_score:
            wins += 1
            if wins == 100:
                print(f'Ding Ding Ding, Episode: {episode-100} - {episode}')
                break
        else:
            wins = 0        

    env.close()

# q-learning
Q learning is a model free, off policy reinforcement learning algorithm that learns a policy to solve finite markov decision processes. Q is a dictionary, mapping the actions taken at states to expected reward. Expected rewards are updated every time an action is taken. Actions are chosen based on Q, in this, I used the epsilon-greedy action selection policy.

Q is learned via temporal difference learning. The value of a given state action pair in Q is equal to the earned reward of taking the action in state and the discounted value of the expected reward for taking the best move in the next states. Future states are discounted by gamma^(number of episodes next state is in the future). gamma is commonly regarded as the probability to survive to the next time step. This means that with a high gamma, actions that will put the agent in line to earn lots of reward in the future will have a higher expected reward.
```
Q[state][action] = (1-α)Q[state][action] + α(reward+γ*max(Q[next_state]))
```

The learning rate is a measure of how much new observations are worth. At a learning rate of alpha=1, all previously learned values are forgotten.
```
Q[state][action] = reward + γ*max(Q[next_state])
```

The epsilon-greedy action selection policy greedily chooses the next action based on maximum expected reward, but, randomly chooses the next action epsilon/100 times.
```
action = argmax(q[state]) if random() > epsilon else action_space.sample()
```

Reinforcment learning consists of exploring the enviornment to gain exposure and exploiting this knowledge to maximize utility. So, I set up the Q learning algorithm to work in two phases: exploration and exploitation. The learning stats with rapid exploration and then switches to slow parameter tuning of actions found to be valuable.
```
alpha = lambda episode: .1 if episode > 200 else .6
epsilon = lambda episode: .1 if episode > 200 else 1
```


## More
[Q Learning Algorithm](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html)


[Action Selection Policies](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html#aselection)


[How to discretize. Gamma, alpha and epsilon parameters](https://dev.to/n1try/cartpole-with-q-learning---first-experiences-with-openai-gym)


[Q-Learning](https://en.wikipedia.org/wiki/Q-learning)


[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)

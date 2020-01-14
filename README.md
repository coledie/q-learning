# q-learning
Q learning is a model free, off policy reinforcement learning algorithm that learns the expected rewards for actions to be taken, thus solving finite markov decision processes. Q is a dictionary, mapping the actions taken at states to expected reward. Expected rewards are updated every time an action is taken via a temporal difference learning rule. Action selection is done based off of the expected reward values, given by the Q table.

Q Learning is a model free reinforcement learning algorithm, this is when the reward function is more or less memorized by the agent repeatedly observing the reward recieved after taking certain actions in certain states. On the other hand, model based reinforcement learning is when the agent is attempting to understand the underlying mechanics of the game they are playing, allowing them to plan ahead and generalize their experiences.

The Temporal Difference Rule measures how much value there is to taking a certain action now, based on what it can lead to in the future. The only parameter of the TD rule is lambda, the discount factor, which is how much the value of a future state is discounted for each timestep it is away from the current state.
For example, consider the scenario where a pirate can choose to turn left or right after undocking. If he turns left he may will find a treasure trove in 3 months time, but if he turns right he will find a single coin on the ground in less than a day. TD(lambda=0) would capture very little information about the future worth of actions, thus enticing the pirate to take a right and recieve only a coin. On the other hand, a TD(lambda=1) rule would capture all future potential of both possible actions, leading the pirate to the treasure.

Q Learning uses the TD rule to update the expected rewards table, with enough iterations over all state-action pairings, the Q table should serve to accuractely tell the agent how to achieve the most reward.
```
Q[state][action] = (1-α)Q[state][action] + α(reward+γ*max(Q[next_state]))
```

In words this is,
```
The expected reward for taking action A in state S = (1-learning rate)(currently understood expected reward for taking action A in state S) + (learning_rate)(recently observed reward for taking action A in state S + discounted(expected reward for taking the optimal action for the next state))
```

The learning rate(alpha) is a measure of how much new observations are worth. At a learning rate of alpha=1, all previously learned values are forgotten.
```
Q[state][action] = (1-1)Q[state][action] + (1)(reward+γ*max(Q[next_state])) = reward + γ*max(Q[next_state])
```

Learning rate, alpha=0, new observations do not affect the Q table.
```
Q[state][action] = (1-0)Q[state][action] + (0)(reward+γ*max(Q[next_state])) = Q[state][action]
```

The discount factor(lambda) is how much weight is given to expected future action rewards.
Discount factor, lambda = 0, expected reward depends soley on what can be achieved in the next state.
```
Q[state][action] = (1-α)Q[state][action] + α(reward+(0)*max(Q[next_state])) = (1-α)Q[state][action] + α(reward)
```

Discount factor, lambda = 1, expected reward is based on all future actions to be taken.
```
Q[state][action] = (1-α)Q[state][action] + α(reward+(1)*max(Q[next_state])) = (1-α)Q[state][action] + α(reward+max(Q[next_state]))
```

Discount factor, 0 < lambda < 1, the next state is valued at lambda^1(expected reward), the state after that is lambda^2(expected reward)... decaying until terms equal 0.

The epsilon-greedy action selection policy greedily chooses the next action based on maximum expected reward, but, randomly chooses the next action (1 - epsilon)% times.
```
action = argmax(q[state]) if random() > epsilon else action_space.sample()
```

Reinforcment learning consists of exploring the enviornment to gain exposure and exploiting this knowledge to maximize utility. So, I set up the Q learning algorithm to work differently in the two phases with parameters set as follows.
```
alpha = f(episode): .1 if episode > 200 else .6
epsilon = f(episode): .1 if episode > 200 else 1
```

## More
[Q Learning Algorithm](https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html)

[Action Selection Policies](https://www.cse.unsw.edu.au/~cs9417ml/RL1/tdlearning.html#aselection)

[How to discretize. Gamma, alpha and epsilon parameters](https://dev.to/n1try/cartpole-with-q-learning---first-experiences-with-openai-gym)

[Q-Learning](https://en.wikipedia.org/wiki/Q-learning)

[Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

[Temporal Difference Learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)

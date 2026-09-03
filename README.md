# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium
# Reg No:212223240138
# Name: Ranjan Kumar G 
# Reg no:212223240138
## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement the **SARSA (State-Action-Reward-State-Action)** reinforcement learning control algorithm using the Gymnasium `FrozenLake-v1` environment.

The agent must learn an optimal policy by interacting with the environment and updating its Q-values based on the action actually selected in the next state.

The learned Q-table, state-value function, policy, average reward, and learning curve are used to evaluate the performance of the agent.

---

## Software Requirements

* Python 3.x
* Jupyter Notebook / Google Colab
* Gymnasium
* NumPy
* Matplotlib

### Installation

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description

The experiment uses the **FrozenLake-v1** environment provided by Gymnasium.

FrozenLake is a grid-world environment in which the agent starts from a starting state and must reach the goal while avoiding holes.

The environment contains a **4 × 4 grid**:

```text
S F F F
F H F H
F F F H
H F F G
```

Where:

| Symbol | Meaning             |
| ------ | ------------------- |
| `S`    | Starting state      |
| `F`    | Frozen/safe surface |
| `H`    | Hole                |
| `G`    | Goal                |

The environment contains:

* **16 states**
* **4 possible actions**

The actions are:

| Action | Meaning |
| ------ | ------- |
| `0`    | Left    |
| `1`    | Down    |
| `2`    | Right   |
| `3`    | Up      |

For this experiment, `is_slippery=False` is used so that the environment is deterministic and the learning behaviour is easier to observe and reproduce.

The reward structure is:

* `0` for a normal movement
* `0` when the agent falls into a hole
* `1` when the agent reaches the goal

---

## Theory

### SARSA

SARSA stands for:

```text
State → Action → Reward → State → Action
```

It is an **on-policy temporal-difference control algorithm**.

The name SARSA comes from the sequence:

```text
St, At, Rt+1, St+1, At+1
```

SARSA updates the Q-value using the action that is **actually selected by the current policy** in the next state.

The SARSA update rule is:

```text
Q(St, At) ← Q(St, At) +
            α [ Rt+1 + γ Q(St+1, At+1) - Q(St, At) ]
```

Where:

| Symbol   | Meaning                                       |
| -------- | --------------------------------------------- |
| `St`     | Current state                                 |
| `At`     | Current action                                |
| `Rt+1`   | Reward received after taking action `At`      |
| `St+1`   | Next state                                    |
| `At+1`   | Next action selected using the current policy |
| `α`      | Learning rate                                 |
| `γ`      | Discount factor                               |
| `Q(s,a)` | Action-value function                         |

### Important Point

SARSA is an **on-policy** algorithm because it learns the value of the policy that it is currently following.

The next action `At+1` is selected using the same epsilon-greedy policy used by the agent.

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy strategy for selecting actions.

With probability `ε`, the agent explores by selecting a random action.

With probability `1 - ε`, the agent exploits by selecting the action with the highest Q-value.

```text
                 random action       with probability ε
a =
                 argmax Q(s,a)       with probability 1 - ε
```

Initially, epsilon is high so that the agent explores different actions.

As training progresses, epsilon is gradually decreased, allowing the agent to exploit the knowledge learned in the Q-table.

The epsilon values used in this experiment are:

```text
Initial epsilon = 1.0
Minimum epsilon = 0.05
Epsilon decay   = 0.9995
```

---

## Algorithm

1. Create the `FrozenLake-v1` environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate `α`.
4. Set the discount factor `γ`.
5. Initialize epsilon for the epsilon-greedy policy.
6. For every training episode:

   * Reset the environment.
   * Select the initial action using the epsilon-greedy policy.
7. For every step:

   * Execute the selected action.
   * Observe the next state and reward.
   * If the episode has terminated, update the Q-value using the reward.
   * Otherwise, select the next action using the epsilon-greedy policy.
   * Apply the SARSA update rule.
   * Move to the next state and action.
8. Decrease epsilon after each episode.
9. Repeat until all training episodes are completed.
10. Calculate the state-value function from the learned Q-table.
11. Extract the learned policy using the action with the highest Q-value for every state.
12. Calculate the average reward over the last 1000 episodes.
13. Plot the learning curve.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "FFSF",
    "FFFF",
    "FFFF",
    "FGFF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=True
)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05
epsilon_decay = 0.9995


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Select an action using epsilon-greedy strategy.
    """

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    max_q = np.max(Q[state])

    # Select randomly among equally good actions
    best_actions = np.flatnonzero(Q[state] == max_q)

    return np.random.choice(best_actions)


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Select initial action using epsilon-greedy policy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Check whether episode has ended
        done = terminated or truncated

        if done:

            # Terminal state has no future Q-value
            target = reward

            # SARSA update
            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            break

        # Select next action using the same epsilon-greedy policy
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA target
        target = reward + gamma * Q[next_state, next_action]

        # SARSA update
        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        # Move to next state-action pair
        state = next_state
        action = next_action

    # Store reward for this episode
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# Calculate State-Value Function and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)

print_policy(learned_policy)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 3)
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
print("Name:Ranjan Kumar G")
print("Reg No:212223240138")
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")

plt.grid(True)
plt.show()

env.close()
```

---

## Output

## Episodes=10000:

### Final Q-table-1
<img width="327" height="382" alt="image" src="https://github.com/user-attachments/assets/ff40c166-bfd6-4b68-922f-2d27e0225bf6" />


### Estimated State-Value Function

<img width="402" height="130" alt="image" src="https://github.com/user-attachments/assets/fc44bf00-baf7-466d-9ef0-a3c786caf1b8" />

### Learned Policy
<img width="310" height="138" alt="image" src="https://github.com/user-attachments/assets/4f5d0a80-0335-4ecd-9c9d-eedd8ef4008c" />


### Average Reward

<img width="548" height="30" alt="image" src="https://github.com/user-attachments/assets/a2a6c70b-fa69-4b17-a35e-597bf4152d04" />

### Plot Learning Curve
<img width="972" height="637" alt="image" src="https://github.com/user-attachments/assets/6dc26549-0388-49ad-8860-530dbe7e1931" />

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment, and the agent learned an action-value function using the epsilon-greedy policy. The learned Q-table and policy enable the agent to select suitable actions to reach the goal while avoiding holes.

---

## Inference

The experiment shows that SARSA can learn an effective policy through repeated interaction with the environment. The epsilon-greedy strategy provides a balance between exploration and exploitation during learning. The Q-table gradually improves as the agent receives rewards and updates its state-action values.

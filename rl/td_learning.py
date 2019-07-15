import gym
import matplotlib.pyplot as plt
import numpy as np


# Start Environment
env = gym.make("FrozenLake-v0")
# env = gym.make("FrozenLake8x8-v0")
print("Starting environment . . .")

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Parameters (ALPHA: Step Size | GAMMA: Discount Factor)
num_episodes = 75000
max_steps_per_episode = 750

ALPHA = 0.1
GAMMA = 0.99
EPS_MAX = 1
EPS_MIN = 0.01
EPS_DECAY = 0.0001

##########################
#  Q-Learning Algorithm  #
##########################

q_table = np.random.random_sample((state_space_size, action_space_size))

if state_space_size == 16:
    q_table[5] = 0
    q_table[7] = 0
    q_table[11] = 0
    q_table[12] = 0
    q_table[15] = 0
else:
    q_table[19] = 0
    q_table[29] = 0
    q_table[35] = 0
    q_table[41] = 0
    q_table[42] = 0
    q_table[46] = 0
    q_table[49] = 0
    q_table[52] = 0
    q_table[54] = 0
    q_table[59] = 0
    q_table[63] = 0

EPS = 1

rewards_all_episodes = []
for episode in range(num_episodes):
    state = env.reset()
    done = False

    for step in range(max_steps_per_episode):
        EPS_threshold = np.random.random_sample()

        if EPS_threshold > EPS:
            # Exploit
            action = np.argmax(q_table[state])
        else:
            # Explore
            action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)

        q_table[state, action] = (1 - ALPHA)*q_table[state, action] + ALPHA*(reward + GAMMA*np.max(q_table[next_state]))

        state = next_state

        if done:
            rewards_all_episodes.append(reward)
            break

    EPS = EPS_MIN + (EPS_MAX - EPS_MIN)*np.exp(-EPS_DECAY*episode)

temp_result = []
rewards_per_100_episodes = np.split(np.array(rewards_all_episodes), num_episodes/100)
for r in rewards_per_100_episodes:
    temp_result.append(sum(r)/100)
q_learning_result = np.array(temp_result)
print("Q-Learning completed !")

#####################
#  Sarsa Algorithm  #
#####################

sarsa_table = np.random.random_sample((state_space_size, action_space_size))

if state_space_size == 16:
    sarsa_table[5] = 0
    sarsa_table[7] = 0
    sarsa_table[11] = 0
    sarsa_table[12] = 0
    sarsa_table[15] = 0
else:
    sarsa_table[19] = 0
    sarsa_table[29] = 0
    sarsa_table[35] = 0
    sarsa_table[41] = 0
    sarsa_table[42] = 0
    sarsa_table[46] = 0
    sarsa_table[49] = 0
    sarsa_table[52] = 0
    sarsa_table[54] = 0
    sarsa_table[59] = 0
    sarsa_table[63] = 0

EPS = 1

rewards_all_episodes = []
for episode in range(num_episodes):
    state = env.reset()
    done = False

    EPS_threshold = np.random.random_sample()
    if EPS_threshold > EPS:
        # Exploit
        action = np.argmax(sarsa_table[state])
    else:
        # Explore
        action = env.action_space.sample()

    for step in range(max_steps_per_episode):
        next_state, reward, done, info = env.step(action)

        if done:
            rewards_all_episodes.append(reward)
            sarsa_table[state, action] = (1 - ALPHA)*sarsa_table[state, action] + ALPHA*reward
            break

        if EPS_threshold > EPS:
            # Exploit
            next_action = np.argmax(sarsa_table[next_state])
        else:
            # Explore
            next_action = env.action_space.sample()

        sarsa_table[state, action] = (1 - ALPHA)*sarsa_table[state, action] + ALPHA*(reward + GAMMA*sarsa_table[next_state, next_action])
        state, action = next_state, next_action

    EPS = EPS_MIN + (EPS_MAX - EPS_MIN)*np.exp(-EPS_DECAY*episode)

temp_result = []
rewards_per_100_episodes = np.split(np.array(rewards_all_episodes), num_episodes/100)
for r in rewards_per_100_episodes:
    temp_result.append(sum(r)/100)
sarsa_result = np.array(temp_result)
print("Sarsa completed !")

# Plot Graphs Using Matplotlib
plt.plot(q_learning_result, color="red", label="Q-Learning")
plt.plot(sarsa_result, color="green", label="Sarsa")
plt.title("FrozenLake - Temporal-Difference Learning")
plt.xlabel("Number of Episodes (x100)")
plt.ylabel("Success Rate")

plt.legend()
plt.show()

# End Environment
env.close()
print("Environment ended !")

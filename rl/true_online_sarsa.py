import gym
import matplotlib.pyplot as plt
import numpy as np


# Start Environment
env = gym.make("FrozenLake-v0")
# env = gym.make("FrozenLake8x8-v0")
print("Starting environment . . .")

state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Parameters (GAMMA: Discount Factor)
num_episodes = 75000
max_steps_per_episode = 750

ALPHA = 0.1
GAMMA = 0.99
EPS_MAX = 1
EPS_MIN = 0.01
EPS_DECAY = 0.0001

#######################
#  True online Sarsa  #
#######################

_done_flag = False

def feature_function(s, a):
    x = np.zeros(state_space_size*action_space_size)
    if _done_flag:
        return x
    else:
        table = np.reshape(x, (state_space_size, action_space_size))
        table[s][a] = 1
        result = table.flatten()
        return result

def true_online_sarsa(LAMBDA):
    global _done_flag

    EPS = 1
    w = np.zeros(state_space_size*action_space_size)

    rewards_all_episodes = []
    for episode in range(num_episodes):
        state = env.reset()
        _done_flag = False

        EPS_threshold = np.random.random_sample()
        if EPS_threshold > EPS:
            # Exploit
            max_q_value = -1024
            action = -1

            for i in range(action_space_size):
                q_value = np.dot(w, feature_function(state, i))
                if q_value > max_q_value:
                    max_q_value = q_value
                    action = i
        else:
            # Explore
            action = env.action_space.sample()

        x = feature_function(state, action)
        z = np.zeros(state_space_size*action_space_size)
        old_q = 0

        for step in range(max_steps_per_episode):
            next_state, reward, _done_flag, info = env.step(action)

            if EPS_threshold > EPS:
                # Exploit
                max_q_value = -1024
                next_action = -1

                for i in range(action_space_size):
                    q_value = np.dot(w, feature_function(next_state, i))
                    if q_value > max_q_value:
                        max_q_value = q_value
                        next_action = i
            else:
                # Explore
                next_action = env.action_space.sample()

            next_x = feature_function(next_state, next_action)
            q = np.dot(w, x)
            next_q = np.dot(w, next_x)
            delta = reward + GAMMA*next_q - q
            z = GAMMA*LAMBDA*z + (1 - ALPHA*GAMMA*LAMBDA*np.dot(z, x))*x
            w = w + ALPHA*(delta + q - old_q)*z - ALPHA*(q - old_q)*x
            old_q, x, action = next_q, next_x, next_action

            if _done_flag:
                rewards_all_episodes.append(reward)
                break

        EPS = EPS_MIN + (EPS_MAX - EPS_MIN)*np.exp(-EPS_DECAY*episode)

    temp_result = []
    rewards_per_100_episodes = np.split(np.array(rewards_all_episodes), num_episodes/100)
    for r in rewards_per_100_episodes:
        temp_result.append(sum(r)/100)
    true_online_sarsa_result = np.array(temp_result)
    return true_online_sarsa_result

# Result
result1 = true_online_sarsa(0.9)
print("33.3% completed . . .")
result2 = true_online_sarsa(0.6)
print("66.6% completed . . .")
result3 = true_online_sarsa(0.3)
print("99.9% completed . . .")

# Plot Graphs Using Matplotlib
plt.plot(result1, color="red", label="lambda=0.9")
plt.plot(result2, color="green", label="lambda=0.6")
plt.plot(result3, color="blue", label="lambda=0.3")
plt.title("FrozenLake - True online Sarsa")
plt.xlabel("Number of Episodes (x100)")
plt.ylabel("Success Rate")

plt.legend()
plt.show()

# End Environment
env.close()
print("Environment ended !")

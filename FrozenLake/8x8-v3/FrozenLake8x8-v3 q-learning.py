import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# Register FrozenLake with is_slippery False
register(
    id= 'FrozenLake8x8-v3',
    entry_point= 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8','is_slippery': False}
)
env = gym.make('FrozenLake8x8-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Discount fator
dis = 0.9
num_episodes = 4000

# Create lists to contain total rewards and steps per episode 
rList = []

for i in range(num_episodes):
    #e = 1. / ((i//1000)+1) # decaying e-greedy

    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-table learning algorithm
    while not done:
        action = np.argmax(Q[state,:]+np.random.randn(1, env.action_space.n) / (i+1))
        #if np.random.rand(1) < e:
        #    action = env.action_space.sample()
        #else:
        #    action = np.argmax(Q[state,:])

        new_state, reward, done,_ = env.step(action)

        Q[state, action] = reward + dis*np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: "+ str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

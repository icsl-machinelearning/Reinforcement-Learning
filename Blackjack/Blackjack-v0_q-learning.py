import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')

# Initialize table with all zeros
Q = np.zeros([32, 11, 2, env.action_space.n])
# Discount fator
dis = 0.95
num_episodes = 10000
learning_rate = 0.1

# Create lists to contain total rewards and steps per episode 
rList = []

for i in range(num_episodes):
    e = 1. / ((i//100)+1) # decaying e-greedy

    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-table learning algorithm
    while not done:
        #action = np.argmax(Q[state[0],state[1], state[2],:]+np.random.randn(1, env.action_space.n) / (i+1))
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state[0], state[1], int(state[2]),:])
        
        new_state, reward, done,_ = env.step(action)

        Q[state[0], state[1], int(state[2]), action] = (1-learning_rate)*Q[state[0], state[1], int(state[2]), action]+learning_rate*(reward + dis*np.max(Q[new_state[0], new_state[1], int(new_state[2]),:]))

        rAll += reward
        state = new_state

    rList.append(rAll)
print("Win rate: "+ str(rList.count(1)/num_episodes), "Draw rate: "+ str(rList.count(0)/num_episodes), "Lose rate: "+ str(rList.count(-1)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

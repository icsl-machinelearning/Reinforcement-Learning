import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')

# Initialize table with all zeros
Q = np.zeros([32, 11, 2, env.action_space.n])
# Discount fator
num_episodes = 8000

# Create lists to contain total rewards and steps per episode 
rList = []

for i in range(num_episodes):



    #e = 1. / ((i//100)+1) # decaying e-greedy

    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-table learning algorithm
    while not done:
        action = env.action_space.sample()
        
        new_state, reward, done,_ = env.step(action)

        rAll += reward
        state = new_state

    rList.append(rAll)
print("Win rate: "+ str(rList.count(1)/num_episodes), "Draw rate: "+ str(rList.count(0)/num_episodes), "Lose rate: "+ str(rList.count(-1)/num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()

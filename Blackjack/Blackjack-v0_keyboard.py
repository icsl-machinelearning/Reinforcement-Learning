import gym
import readchar

#MACROS
STICK = 0
HIT = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {
    '\x1b[A': UP,
    '\x1b[B': HIT,
    '\x1b[C': RIGHT,
    '\x1b[D': STICK
}

env = gym.make('Blackjack-v0')
state = env.reset()
print(env._get_obs())

while True:
    # Choose an action from keyboard
    key = readchar.readkey()
    if key not in arrow_keys.keys():
        print("Game aborted!")
        break

    action = arrow_keys[key]
    state, reward, done, info = env.step(action)
    print(env._get_obs())  # Show the board after action
    print("State: ", state, "Action: ", action, "Reward: ", reward, "Info: ", info)

    if done:
        print("Finished with reward", reward)
        break
    

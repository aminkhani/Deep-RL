# Q-Learning with FrozenLake 4x4

![alt text](http://simoninithomas.com/drlc/Qlearning/frozenlake4x4.png)




```python
import numpy as np
import gym
import random
import time
```

## Step 1: Create the environment 




```python
env = gym.make("FrozenLake-v1", render_mode='human')
```

    c:\Users\Amin\AppData\Local\Programs\Python\Python310\lib\site-packages\gym\core.py:329: DeprecationWarning: [33mWARN: Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(
    c:\Users\Amin\AppData\Local\Programs\Python\Python310\lib\site-packages\gym\wrappers\step_api_compatibility.py:39: DeprecationWarning: [33mWARN: Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.[0m
      deprecation(
    

## Step 2: Create the Q-table and initialize it 



```python
action_size = env.action_space.n
state_size = env.observation_space.n
print("Action Size: " , action_size, "(0: LEFT - 1: DOWN - 2: RIGHT - 3: UP)","\nState Size: ", state_size)
```

    Action Size:  4 (0: LEFT - 1: DOWN - 2: RIGHT - 3: UP) 
    State Size:  16
    


```python
# Create our Q table with state_size rows and action_size columns (
qtable = np.zeros((state_size, action_size))
print("Each row is State & Each column is Action\n------------------\n", qtable)
```

    Each row is State & Each column is Action
    ------------------
     [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    

## Step 3: Create the hyperparameters 



```python
total_episodes = 20000       # Total episodes
learning_rate = 0.7          # Learning rate
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.005            # Exponential decay rate for exploration prob
```

## Step 4: The Q-learning algorithm 
  ![alt text](http://simoninithomas.com/drlc/Qlearning//qtable_algo.png)



```python
# 1. List of rewards
rewards = []

# 2. Until learning is stopped
Start = time.time()
for episode in range(100):
    print(f"\n---------- Episode: {episode} ----------\n")
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        print(f"Step: {step}")
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
            #print(exp_exp_tradeoff, "action", action)

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
            #print("action random", action)
            
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True and reward == 1: 
            print("WIN")
            break
        if done == True: 
            print("Dead")
            break
         
    # Reduce epsilon (because we need less and less exploration) # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) # https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/ 
    rewards.append(total_rewards)
    
End = time.time()
Time = End - Start
print ("Score over time: ",  sum(rewards)/total_episodes)
print('Execution time: {:.3f}'.format(Time), 'seconds')
float_formatter = "{:.9f}".format # https://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array
np.set_printoptions(formatter={'float_kind': float_formatter})
print("\n------------------------------------------\n")
print(qtable)
```

    
    ---------- Episode: 0 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 1 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Dead
    
    ---------- Episode: 2 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 3 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 4 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 5 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 6 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Dead
    
    ---------- Episode: 7 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 8 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 9 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Dead
    
    ---------- Episode: 10 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 11 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    WIN
    
    ---------- Episode: 12 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 13 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Dead
    
    ---------- Episode: 14 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Dead
    
    ---------- Episode: 15 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 16 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 17 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 18 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 19 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Dead
    
    ---------- Episode: 20 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 21 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 22 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 23 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 24 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 25 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 26 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 27 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 28 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 29 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Dead
    
    ---------- Episode: 30 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 31 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 32 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Dead
    
    ---------- Episode: 33 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 34 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 35 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Dead
    
    ---------- Episode: 36 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 37 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 38 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 39 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Dead
    
    ---------- Episode: 40 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 41 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 42 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 43 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Dead
    
    ---------- Episode: 44 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 45 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 46 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 47 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 48 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 49 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Dead
    
    ---------- Episode: 50 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 51 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 52 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 53 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Dead
    
    ---------- Episode: 54 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 55 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 56 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 57 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 58 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Dead
    
    ---------- Episode: 59 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 60 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 61 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 62 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 63 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 64 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Dead
    
    ---------- Episode: 65 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 66 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    Dead
    
    ---------- Episode: 67 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 68 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 69 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    WIN
    
    ---------- Episode: 70 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Dead
    
    ---------- Episode: 71 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 72 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Dead
    
    ---------- Episode: 73 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Dead
    
    ---------- Episode: 74 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 75 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Dead
    
    ---------- Episode: 76 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 77 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 78 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Dead
    
    ---------- Episode: 79 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Dead
    
    ---------- Episode: 80 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 81 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Dead
    
    ---------- Episode: 82 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 83 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 84 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 85 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Dead
    
    ---------- Episode: 86 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Dead
    
    ---------- Episode: 87 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    Step: 17
    Step: 18
    Step: 19
    
    ---------- Episode: 88 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Dead
    
    ---------- Episode: 89 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 90 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 91 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 92 ----------
    
    Step: 0
    Step: 1
    Dead
    
    ---------- Episode: 93 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Dead
    
    ---------- Episode: 94 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Step: 12
    Step: 13
    Step: 14
    Step: 15
    Step: 16
    WIN
    
    ---------- Episode: 95 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Dead
    
    ---------- Episode: 96 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 97 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    
    ---------- Episode: 98 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Step: 8
    Step: 9
    Step: 10
    Step: 11
    Dead
    
    ---------- Episode: 99 ----------
    
    Step: 0
    Step: 1
    Step: 2
    Step: 3
    Step: 4
    Step: 5
    Step: 6
    Step: 7
    Dead
    Score over time:  0.00015
    Execution time: 222.024 seconds
    
    ------------------------------------------
    
    [[0.116254266 0.164091572 0.132325925 0.120685028]
     [0.115975173 0.091025689 0.029127771 0.147197861]
     [0.129772799 0.123516260 0.142299227 0.137636273]
     [0.137416588 0.043938467 0.034260854 0.141293096]
     [0.227130758 0.175571922 0.130245606 0.126834589]
     [0.000000000 0.000000000 0.000000000 0.000000000]
     [0.125870505 0.008920420 0.029591406 0.031704682]
     [0.000000000 0.000000000 0.000000000 0.000000000]
     [0.097892421 0.082213038 0.039687204 0.236009251]
     [0.298722987 0.392797704 0.000000000 0.027622910]
     [0.061230785 0.016326329 0.509032351 0.094625192]
     [0.000000000 0.000000000 0.000000000 0.000000000]
     [0.000000000 0.000000000 0.000000000 0.000000000]
     [0.198650787 0.449207500 0.000000000 0.000000000]
     [0.309557500 0.895557250 0.547417675 0.612424750]
     [0.000000000 0.000000000 0.000000000 0.000000000]]
    

## Step 5: Use our Q-table to play FrozenLake ! 



```python
env.reset()
max_steps = 99
n = 100
win = 0
loss = 0
for episode in range(n):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE: ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            if new_state == 15:
                win += 1
                print(">> We reached our Goal")
            else:
                loss += 1
                print(">> We fell into a hole")
            
            # We print the number of step it took.
            print("Number of steps: ", step)
            
            break
        state = new_state
env.close()
print("****************************************************")
print(f"\nTotal Episode: {n}\nMax Step per Episode: {max_steps}\nWin: {win}\nLoss: {loss}")
```

    ****************************************************
    EPISODE:  0
    >> We fell into a hole
    Number of steps:  23
    ****************************************************
    EPISODE:  1
    >> We fell into a hole
    Number of steps:  17
    ****************************************************
    EPISODE:  2
    >> We fell into a hole
    Number of steps:  33
    ****************************************************
    EPISODE:  3
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  4
    >> We fell into a hole
    Number of steps:  18
    ****************************************************
    EPISODE:  5
    >> We reached our Goal
    Number of steps:  8
    ****************************************************
    EPISODE:  6
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  7
    >> We reached our Goal
    Number of steps:  16
    ****************************************************
    EPISODE:  8
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  9
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  10
    >> We fell into a hole
    Number of steps:  15
    ****************************************************
    EPISODE:  11
    >> We fell into a hole
    Number of steps:  24
    ****************************************************
    EPISODE:  12
    >> We reached our Goal
    Number of steps:  14
    ****************************************************
    EPISODE:  13
    >> We fell into a hole
    Number of steps:  22
    ****************************************************
    EPISODE:  14
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  15
    >> We reached our Goal
    Number of steps:  8
    ****************************************************
    EPISODE:  16
    >> We fell into a hole
    Number of steps:  12
    ****************************************************
    EPISODE:  17
    >> We fell into a hole
    Number of steps:  17
    ****************************************************
    EPISODE:  18
    >> We fell into a hole
    Number of steps:  15
    ****************************************************
    EPISODE:  19
    >> We fell into a hole
    Number of steps:  30
    ****************************************************
    EPISODE:  20
    >> We reached our Goal
    Number of steps:  11
    ****************************************************
    EPISODE:  21
    >> We fell into a hole
    Number of steps:  34
    ****************************************************
    EPISODE:  22
    >> We reached our Goal
    Number of steps:  45
    ****************************************************
    EPISODE:  23
    >> We fell into a hole
    Number of steps:  43
    ****************************************************
    EPISODE:  24
    >> We reached our Goal
    Number of steps:  25
    ****************************************************
    EPISODE:  25
    >> We fell into a hole
    Number of steps:  25
    ****************************************************
    EPISODE:  26
    >> We fell into a hole
    Number of steps:  8
    ****************************************************
    EPISODE:  27
    >> We fell into a hole
    Number of steps:  6
    ****************************************************
    EPISODE:  28
    >> We fell into a hole
    Number of steps:  15
    ****************************************************
    EPISODE:  29
    >> We fell into a hole
    Number of steps:  4
    ****************************************************
    EPISODE:  30
    >> We fell into a hole
    Number of steps:  5
    ****************************************************
    EPISODE:  31
    >> We fell into a hole
    Number of steps:  42
    ****************************************************
    EPISODE:  32
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  33
    >> We fell into a hole
    Number of steps:  25
    ****************************************************
    EPISODE:  34
    >> We fell into a hole
    Number of steps:  25
    ****************************************************
    EPISODE:  35
    >> We fell into a hole
    Number of steps:  7
    ****************************************************
    EPISODE:  36
    >> We fell into a hole
    Number of steps:  16
    ****************************************************
    EPISODE:  37
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  38
    >> We fell into a hole
    Number of steps:  18
    ****************************************************
    EPISODE:  39
    >> We fell into a hole
    Number of steps:  20
    ****************************************************
    EPISODE:  40
    >> We fell into a hole
    Number of steps:  47
    ****************************************************
    EPISODE:  41
    >> We reached our Goal
    Number of steps:  31
    ****************************************************
    EPISODE:  42
    >> We reached our Goal
    Number of steps:  9
    ****************************************************
    EPISODE:  43
    >> We fell into a hole
    Number of steps:  16
    ****************************************************
    EPISODE:  44
    >> We fell into a hole
    Number of steps:  27
    ****************************************************
    EPISODE:  45
    >> We reached our Goal
    Number of steps:  7
    ****************************************************
    EPISODE:  46
    >> We fell into a hole
    Number of steps:  8
    ****************************************************
    EPISODE:  47
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  48
    >> We fell into a hole
    Number of steps:  19
    ****************************************************
    EPISODE:  49
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  50
    >> We fell into a hole
    Number of steps:  10
    ****************************************************
    EPISODE:  51
    >> We fell into a hole
    Number of steps:  14
    ****************************************************
    EPISODE:  52
    >> We reached our Goal
    Number of steps:  11
    ****************************************************
    EPISODE:  53
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  54
    >> We fell into a hole
    Number of steps:  8
    ****************************************************
    EPISODE:  55
    >> We reached our Goal
    Number of steps:  15
    ****************************************************
    EPISODE:  56
    >> We fell into a hole
    Number of steps:  8
    ****************************************************
    EPISODE:  57
    >> We reached our Goal
    Number of steps:  22
    ****************************************************
    EPISODE:  58
    >> We fell into a hole
    Number of steps:  5
    ****************************************************
    EPISODE:  59
    >> We reached our Goal
    Number of steps:  13
    ****************************************************
    EPISODE:  60
    >> We fell into a hole
    Number of steps:  32
    ****************************************************
    EPISODE:  61
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  62
    >> We fell into a hole
    Number of steps:  8
    ****************************************************
    EPISODE:  63
    >> We reached our Goal
    Number of steps:  14
    ****************************************************
    EPISODE:  64
    >> We fell into a hole
    Number of steps:  28
    ****************************************************
    EPISODE:  65
    >> We fell into a hole
    Number of steps:  10
    ****************************************************
    EPISODE:  66
    >> We fell into a hole
    Number of steps:  10
    ****************************************************
    EPISODE:  67
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  68
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  69
    >> We fell into a hole
    Number of steps:  13
    ****************************************************
    EPISODE:  70
    >> We fell into a hole
    Number of steps:  6
    ****************************************************
    EPISODE:  71
    >> We fell into a hole
    Number of steps:  29
    ****************************************************
    EPISODE:  72
    >> We fell into a hole
    Number of steps:  12
    ****************************************************
    EPISODE:  73
    >> We fell into a hole
    Number of steps:  15
    ****************************************************
    EPISODE:  74
    >> We fell into a hole
    Number of steps:  6
    ****************************************************
    EPISODE:  75
    >> We reached our Goal
    Number of steps:  27
    ****************************************************
    EPISODE:  76
    >> We reached our Goal
    Number of steps:  29
    ****************************************************
    EPISODE:  77
    >> We fell into a hole
    Number of steps:  31
    ****************************************************
    EPISODE:  78
    >> We fell into a hole
    Number of steps:  7
    ****************************************************
    EPISODE:  79
    >> We fell into a hole
    Number of steps:  3
    ****************************************************
    EPISODE:  80
    >> We fell into a hole
    Number of steps:  35
    ****************************************************
    EPISODE:  81
    >> We fell into a hole
    Number of steps:  18
    ****************************************************
    EPISODE:  82
    >> We fell into a hole
    Number of steps:  14
    ****************************************************
    EPISODE:  83
    >> We fell into a hole
    Number of steps:  18
    ****************************************************
    EPISODE:  84
    >> We fell into a hole
    Number of steps:  10
    ****************************************************
    EPISODE:  85
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  86
    >> We fell into a hole
    Number of steps:  9
    ****************************************************
    EPISODE:  87
    >> We fell into a hole
    Number of steps:  36
    ****************************************************
    EPISODE:  88
    >> We fell into a hole
    Number of steps:  3
    ****************************************************
    EPISODE:  89
    >> We fell into a hole
    Number of steps:  18
    ****************************************************
    EPISODE:  90
    >> We fell into a hole
    Number of steps:  46
    ****************************************************
    EPISODE:  91
    >> We fell into a hole
    Number of steps:  11
    ****************************************************
    EPISODE:  92
    >> We fell into a hole
    Number of steps:  30
    ****************************************************
    EPISODE:  93
    >> We fell into a hole
    Number of steps:  12
    ****************************************************
    EPISODE:  94
    >> We fell into a hole
    Number of steps:  5
    ****************************************************
    EPISODE:  95
    >> We fell into a hole
    Number of steps:  20
    ****************************************************
    EPISODE:  96
    >> We fell into a hole
    Number of steps:  16
    ****************************************************
    EPISODE:  97
    >> We fell into a hole
    Number of steps:  21
    ****************************************************
    EPISODE:  98
    >> We fell into a hole
    Number of steps:  33
    ****************************************************
    EPISODE:  99
    >> We fell into a hole
    Number of steps:  15
    ****************************************************
    
    Total Episode: 100
    Max Step per Episode: 99
    Win: 17
    Loss: 83
    

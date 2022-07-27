<div class="cell markdown" id="WpABiUpzhVOZ">

## Space Invaders With DQN

</div>

<div class="cell markdown" id="zIyhgtTrhfzh">

## Step 0

1- For using Space Invaders in Colab we need to download ROMS and Import
the environment

2- To displaye the Agent & Environment we use env.render() but not work
in colab, for runing env.render() use the below cells

</div>

<div class="cell code" id="zdAcCZ4RZik9">

``` python
# 1- Run this cell to Import the environment
! wget http://www.atarimania.com/roms/Roms.rar
! mkdir /content/ROM/
! unrar e /content/Roms.rar /content/ROM/
! python -m atari_py.import_roms /content/ROM/
```

</div>

<div class="cell code" id="kNioA4kYfD28">

``` python
# 2- Dowload and install requirements
!pip install gym pyvirtualdisplay > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg > /dev/null 2>&1
!apt-get update > /dev/null 2>&1
!apt-get install cmake > /dev/null 2>&1
!pip install --upgrade setuptools 2>&1
!pip install ez_setup > /dev/null 2>&1
!pip install gym[atari] > /dev/null 2>&1
```

</div>

<div class="cell code" data-execution_count="1" id="3UErC4-9fH27">

``` python
# Next, we define the functions used to show the video by adding it to the CoLab 
import gym
from gym.wrappers import Monitor
import glob
import io
import base64
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay

display = Display(visible=0, size=(1400, 900))
display.start()

"""
Utility functions to enable video recording of gym environment 
and displaying it.
To enable video, just do "env = wrap_env(env)""
"""


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env
```

</div>

<div class="cell markdown" id="LN0nZwyMGadB">

## Step 1: Import the libraries

</div>

<div class="cell code" data-execution_count="2" id="RF19XeI0V4DN">

``` python
import time
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(30)
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
```

</div>

<div class="cell code" data-execution_count="3" id="pg4m4pysV4DQ">

``` python
# Local Libraries
from dqn_agent import DQNAgent
from dqn_cnn import DQNCnn
from stack_frame import preprocess_frame, stack_frame
```

</div>

<div class="cell markdown" id="tfo8jleHGadK">

## Step 2: Create our environment

Initialize the environment in the code cell below.

</div>

<div class="cell code" data-execution_count="4" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="t4NLWW8HV4DR" data-outputId="80231f09-3071-446d-a37c-93563543416b">

``` python
env = wrap_env(gym.make('SpaceInvaders-v0'))
env.seed(0)
```

<div class="output execute_result" data-execution_count="4">

    [0, 592379725]

</div>

</div>

<div class="cell code" data-execution_count="5" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="lnYIMwuEV4DS" data-outputId="434ac1e1-0c4f-4af5-9b62-21803b02dec7">

``` python
# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
```

<div class="output stream stdout">

    Device:  cuda

</div>

</div>

<div class="cell markdown" id="nS221MgXGadP">

## Step 3: Viewing our Enviroment

</div>

<div class="cell code" data-execution_count="6" data-colab="{&quot;height&quot;:350,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="u7InfF_kV4DU" data-outputId="2ade6f06-05d1-4e7e-d0b4-1c7f57eb5576">

``` python
print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)
print("\n")
env.reset()
plt.figure()
plt.imshow(env.reset())
plt.title('Original Frame')
plt.show()
```

<div class="output stream stdout">

``` 
The size of frame is:  (210, 160, 3)
No. of Actions:  6


```

</div>

<div class="output display_data">

![](ffdb27c9be8358fd85fb1b3d58d4d3a05e041612.png)

</div>

</div>

<div class="cell markdown" id="dQ7JEkuTV4DV">

### Execute the code cell below to play with a random policy.

</div>

<div class="cell code" data-execution_count="7" data-colab="{&quot;height&quot;:439,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="j0Mv4hxBV4DV" data-outputId="7c16bf91-cb54-4066-b8f6-1933671e6a2b">

``` python
def random_play():
    score = 0
    env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            env.close()
            show_video()
            print("Your Score at end of game is: ", score)
            break
random_play()
```

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

<div class="output stream stdout">

    Your Score at end of game is:  80.0

</div>

</div>

<div class="cell markdown" id="Sr52nmcpGada">

# Step 4: Preprocessing Frame

</div>

<div class="cell code" id="Tm0lcWXfV4DW">

``` python
env.reset()
plt.figure()
plt.imshow(preprocess_frame(env.reset(), (8, -12, -12, 4), 84), cmap="gray")
plt.title('Pre Processed image')
plt.show()
```

</div>

<div class="cell markdown" id="mJMc3HA8Gade">

## Step 5: Stacking Frame

</div>

<div class="cell code" id="mCS0FjePV4DX">

``` python
def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (8, -12, -12, 4), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames
```

</div>

<div class="cell markdown" id="bH22GTfhV4DY">

## Step 6: Creating our Agent

</div>

<div class="cell code" id="kBIKZNalV4DY">

``` python
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = env.action_space.n
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 64        # Update batch size
LR = 0.0001            # learning rate 
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 1       # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started 
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = 100         # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn)
```

</div>

<div class="cell markdown" id="reH8jzUTV4DZ">

## Step 7: Watching untrained agent play

</div>

<div class="cell code" id="XEe1jAfBV4DZ">

``` python
# watch an untrained agent
state = stack_frames(None, env.reset(), True) 
for j in range(200):
    env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = stack_frames(state, next_state, False)
    if done:
        break 
        
env.close()
show_video()
```

</div>

<div class="cell markdown" id="Fl6HKp9GV4DZ">

## Step 8: Loading Agent

Uncomment line to load a pretrained agent

</div>

<div class="cell code" id="WLTlfl00V4DZ">

``` python
start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
```

</div>

<div class="cell markdown" id="MYqtjUgQV4Da">

## Step 9: Train the Agent with DQN

</div>

<div class="cell code" id="WOt-aLHsV4Da">

``` python
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

plt.plot([epsilon_by_epsiode(i) for i in range(1000)])
```

</div>

<div class="cell code" id="3outDx_1V4Da">

``` python
def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        eps = epsilon_by_epsiode(i_episode)
        while True:
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(action)
            score += reward
            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()
    
    return scores
```

</div>

<div class="cell code" id="tDuOnPGVV4Db">

``` python
scores = train(1000)
```

</div>

<div class="cell markdown" id="jANzNpp-V4Db">

## Step 10: Watch a Smart Agent\!

</div>

<div class="cell code" id="Fs7kQ5S2V4Dc">

``` python
score = 0
state = stack_frames(None, env.reset(), True)
while True:
    env.render()
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    score += reward
    state = stack_frames(state, next_state, False)
    if done:
        print("You Final score is:", score)
        break 
env.close()
show_video()
```

</div>

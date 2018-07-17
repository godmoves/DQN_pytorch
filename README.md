# DQN pytorch
A pytorch implement of DQN.

## Dependencies
- python 3  
- pytorch 0.4  
- numpy  
- gym  

Use `pip` to install what you need.

## Usage
Just type `python dqn.py` in Terminal.

## Training
After about 3M frames (100h) of training, the agent can beat the ai in the
original atari game most of the time. (After another 2M frames of training, I
don't find big differences in the average reward. But according to the test,
the agent's performance is much stable and it gets higher winrate against the
ai in the original game.)
![training](image/training.png)

## Test
Finally, I test the agent in 100 games and it wins 98 games:
```
Step 412432 Episode 98/100 Running mean 8.646
```

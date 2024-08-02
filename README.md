# Autonomous-car-obstacle-avoidance

## Overview

This project makes use of Deep Reinforcement Learning to help a simulated car navigate through a map of obstacles that are randomly generated.
Here, Proximal Policy Optimization (PPO) is used. The steering angle and acceleration are treated as two parameters of an action space. The model rewards forward motion (along the horizontal pathway), and gives a negative reward for collisions or negligible motion.

## To Run

**Installation of packages**: Run 'pip install -r requirements.txt' inside an environment to install the needed packages.

### Project files
- **objs.py** : this file contains class definitions for Car and Environment. NOT RUNNABLE AS MAIN.
- **CarNavigationGymEnv.py** : defines a custom python gym environment for being used during training.
- **renderer.py** : this file is where the graphics for displaying the window with the car and the obstacle grid are rendered.
- **train.py** : this file is used to train the model, using a custom python gym environment, with a PPO model, and test its performance.

### Other files
- **simulate.py** : this file contains a simple rendering program that I wrote, initially in mind as the main renderer, but later as a useful testing tool, to see if the agent and environment are working correctly (in the untrained state).
- **train_obsolete.py** : this was previously the train.py file, attempting to use a DQN model, but I was not able to get it to work, and so I abandoned this approach, and went to a simpler, different approach by using the python Gym environment.

## Resources

### Relevant references
- Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations. https://stable-baselines3.readthedocs.io/en/master/
- Proximal Policy Optimization. https://spinningup.openai.com/en/latest/algorithms/ppo.html
- PPO Algorithm. https://medium.com/@danushidk507/ppo-algorithm-3b33195de14a

### Older references I was using earlier (for a DQN based model)
- Deep Q-Learning/Deep Q-Network (DQN) Explained | Python Pytorch Deep Reinforcement Learning. https://www.youtube.com/watch?v=EUrWGTCGzlA
- https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py
- Reinforcement Learning (DQN) Tutorial. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#:~:text=This%20tutorial%20shows%20how%20to,CartPole%2Dv1%20task%20from%20Gymnasium.&text=The%20agent%20has%20to%20decide,attached%20to%20it%20stays%20upright.

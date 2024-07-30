# Autonomous-car-obstacle-avoidance

## Overview

This project makes use of Deep Reinforcement Learning to help a simulated car navigate through a map of obstacles that are randomly generated.
Here, Deep Q-Learning (DQN, for Deep Q-Networks) is used, by considering the action set as finite, through discreet allowed rotations for the car from its current steering angle.

## To Run

**Installation of packages**: Run 'pip install -r requirements.txt' inside an environment to install the needed packages.

- **objs.py** : this file contains class definitions for Car and Environment. NOT RUNNABLE AS MAIN.
- **simulate.py** : this file is the main file that would be meant to be run, where the graphics for displaying the window with the car and the obstacle grid are created.
- **train.py** : this file is used to train the model, and test its performance.

## Resources

- Deep Q-Learning/Deep Q-Network (DQN) Explained | Python Pytorch Deep Reinforcement Learning. https://www.youtube.com/watch?v=EUrWGTCGzlA
- https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_dql.py
- Reinforcement Learning (DQN) Tutorial. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#:~:text=This%20tutorial%20shows%20how%20to,CartPole%2Dv1%20task%20from%20Gymnasium.&text=The%20agent%20has%20to%20decide,attached%20to%20it%20stays%20upright.
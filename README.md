# Deep Reinforcement Learning and Imitation Learning with sim2sim exploration and behavior cloning

The PDF containing the assignment, along with detailed result explanations, will be published soon.

This repository contains the code and models for a "Deep Reinforcement Learning Assignment (University MSc)". The assignment explores deep reinforcement learning (DRL) techniques in sim2sim and imitation learning.

### Simulation of the Ant-v4 model, that was trained using PPO with the following hyperparameters: batch size of 256, gamma value of 0.99, and 900,000 timesteps.
[![Simulation of the Ant-v4 model, that was trained using PPO with the following hyperparameters: batch size of 256, gamma value of 0.99, and 900,000 timesteps.](http://img.youtube.com/vi/aLZdnPR4RDw/0.jpg)](http://www.youtube.com/watch?v=aLZdnPR4RDw "Ant-v4 Gymnasium PPO #AI #simulation")

### Part 1 - SIM2SIM - GENERALIZATION OF TRAINED POLICIES TO DIFFERENT ENVIRONMENT DYNAMICS
Part 1 focuses on generalizing trained policies to different environment dynamics, in this case: torso mass, using the Proximal Policy Optimization (PPO) algorithm. Hyperparameter tuning with Optuna is used for effective exploration. And HuggingFace (2023) state-of-the-art baseline hyperparameters are tested.

### Part 2 - SIM2SIM - IMITATION LEARNING - LEARNING FROM EXPERT DEMONSTRATIONS / BEHAVIOR CLONING
Part 2 investigates behavior cloning (BC) for imitation learning, where a DRL agent is trained on the "Ant-v4" Gym environment using PPO. The impact of expert data and policy network size on BC agent performance is analyzed. The results highlight the importance of sim2sim and imitation learning in training robust policies that generalize well. This work contributes to advancing DRL understanding and offers insights into the optimization and generalization of sim2sim and imitation learning.

<img src="https://github.com/basverkennis/DRL-PPO-sim2sim-imitationlearning/blob/main/logo.jpeg" alt="Tilburg University Logo" width="20%" padding="5%">

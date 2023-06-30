import gymnasium as gym
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
import matplotlib.pyplot as plt

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

rng = np.random.default_rng(0)
NUM_ENVS = 8

def sample_expert_transitions(expert, env_id, min_episodes):
    wrapped_venv = DummyVecEnv([lambda: RolloutInfoWrapper(gym.make(env_id)) for _ in range(NUM_ENVS)])
    wrapped_venv = VecNormalize(wrapped_venv, norm_obs=True, norm_reward=True)
    rollouts = rollout.rollout(
        expert,
        wrapped_venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=min_episodes),
        rng=rng,
    )
    return rollout.flatten_trajectories(rollouts)

if __name__ == '__main__':
    env_id = 'Ant-v4'
    num_experiments = 10  # Number of experiments to run
    num_episodes = [2, 10, 50, 100, 200, 400, 1000]  # Vary the number of expert episodes

    # Initialize lists to store performance results
    experiments = []
    
    env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize.load('env/vec_normalize_ant_expert.pkl', env)
    expert = PPO.load('models/ant_expert.zip', env)
    
    for experiment in range(num_experiments):
        print(f'Experiment: {experiment}')    
        episode_rewards = []
        for num_ep in num_episodes:
        
            # Generate expert demonstrations using Behavior Cloning
            transitions = sample_expert_transitions(expert, env_id, num_ep)
            
            # Train a policy using Behavior Cloning
            bc_trainer = bc.BC(
                observation_space=expert.observation_space,
                action_space=expert.action_space,
                demonstrations=transitions,
                rng=rng,
            )
            reward, _ = evaluate_policy(
                bc_trainer.policy,  # type: ignore[arg-type]
                expert.env,
                n_eval_episodes=2,
                render=False,
            )
            
            # Disable printing during training
            sys.stdout = open(os.devnull, 'w')

            print("Training a policy using Behavior Cloning")  
            bc_trainer.train(n_epochs=100, progress_bar=False)
            
            # Restore standard output
            sys.stdout = sys.__stdout__

            # Evaluate the trained policy
            reward, _ = evaluate_policy(
                bc_trainer.policy,  # type: ignore[arg-type]
                expert.env,
                n_eval_episodes=2,
                render=False,
            )
            print(f"Reward after training: {reward}")
            episode_rewards.append(reward)
        experiments.append(episode_rewards)

    column_avg = np.mean(experiments, axis=0)
    print(f"Mean reward after training and all experiments: {column_avg}")
    # Plot the performance of BC agents as a function of expert data available
    plt.plot(num_episodes, column_avg, marker='o')
    plt.xlabel('Number of Expert Episodes')
    plt.ylabel('Average Reward')
    plt.title('Performance of BC Agents')
    plt.savefig('figures/AverageReward_2.png')
    plt.show()

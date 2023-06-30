import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from matplotlib import patches

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass

    def reset(self, **kwargs):
        self.env.model.body_mass[1] = self.torso_mass
        return self.env.reset(**kwargs)

def test_model(model_path, torso_masses):
    model = PPO.load(model_path)
    results = []
    split_parts = model_path.split('_')
    model_trained = split_parts[-1].split('.')[0]
    for mass in torso_masses:
        env = make_vec_env('Hopper-v4', n_envs=4, seed=0, vec_env_cls=SubprocVecEnv,
                           wrapper_class=ChangeMassWrapper, wrapper_kwargs=dict(torso_mass=mass))
        env = VecNormalize.load(f'env/hopper_v4_vecnormalize_torso_mass_{model_trained}.pkl', env)
        env.training = False
        env.norm_reward = False
        model.set_env(env)
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
        print(f"Trained model torso mass: {model_trained}, tested on torso mass {mass}. Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        results.append([mean_reward, std_reward])
    return results

def plot_results(results, torso_masses):
    colors = ['blue', 'green', 'red']
    light_colors = ['lightblue', 'lightgreen', 'lightcoral'] 
    labels = ['m = 3', 'm = 6', 'm = 9']
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Performance of hopper policies when testing on target domains with different torso masses')
    
    for i, result in enumerate(results):
        result = np.array(result)
        mean_rewards = result[:, 0]
        std_rewards = result[:, 1]
        # break
        # Create lighter fillings between mean and std lines
        fill_color = light_colors[i]
        axs[i].fill_between(torso_masses, mean_rewards - std_rewards, mean_rewards + std_rewards, color=fill_color)

        # Plot mean rewards with std line
        axs[i].plot(torso_masses, mean_rewards, color=colors[i], label=labels[i])
        axs[i].fill_between(torso_masses, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0)
        
        axs[i].set_xlabel('Torso Mass (kg)')
        axs[i].set_ylabel('Total Reward')
        axs[i].set_title(f'Model - {labels[i]}')
        axs[i].legend(handles=[patches.Patch(color=colors[i])], labels=[labels[i]])
    
    plt.savefig('figures/TotalReward.png')
    plt.show()

if __name__ == "__main__":
    # Train the models
    torso_masses = [3, 6, 9]  # List of torso masses for training
    
    models = []
    for mass in torso_masses:
        models.append(f'models/hopper_torso_mass_{mass}.zip')

    # Test the models
    torso_masses = [3, 4, 5, 6, 7, 8, 9]  # List of torso masses for evaluation environments
    results = []
    for model_path in models:
        model_results = test_model(model_path, torso_masses)
        results.append(model_results)

    # Plot the results
    plot_results(results, torso_masses)

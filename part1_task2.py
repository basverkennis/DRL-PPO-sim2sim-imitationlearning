import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import pickle

# Define the custom wrapper to change torso mass
class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
    
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass

# for Optuna hyperparameter optimization
# def train_expert(env, seed, best_params):      
def train_expert(env, seed):
    print("Training an expert.")
    
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=seed,
        verbose=0,
        batch_size=256,
        gamma=0.99,
        ## for Optuna hyperparameter optimization
        # batch_size=best_params['n_steps']*4, # n_steps * n_envs
        # ent_coef=best_params['ent_coef'],
        # learning_rate=best_params['learning_rate'],
        # n_epochs=best_params['n_epochs'],
        # n_steps=best_params['n_steps'],
        # gamma=best_params['gamma'],
        # gae_lambda=best_params['gae_lambda'],
        # clip_range=best_params['clip_range'],
        # vf_coef=best_params['vf_coef'],
    )
    expert.learn(total_timesteps=5000000)  # Train with the best hyperparameters
    
    return expert

if __name__ == "__main__":
    env_id = 'Hopper-v4'
    # Create the vectorized environment with custom torso masses
    n_envs = 4  # Number of parallel environments
    torso_masses = [3,6,9]  # List of torso masses for evaluation environments
    random_seeds = [0, 1, 2, 3, 4]  # List of random seeds
    
    for torso_mass in torso_masses:
        print(f'Torso mass: {torso_mass}')
        results = []
        models = []
        for seed in random_seeds:
            env = make_vec_env(env_id, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                            wrapper_class=ChangeMassWrapper, wrapper_kwargs=dict(torso_mass=torso_mass))
            
            # Normalize observations and rewards
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

            # for Optuna hyperparameter optimization
            # with open('env/best_params_task1.pkl', 'rb') as f:
            #     best_params = pickle.load(f)
                
            # expert = train_expert(env, seed, best_params)
            expert = train_expert(env, seed)
            
            models.append(expert)
            
            mean_reward, _ = evaluate_policy(expert, env, n_eval_episodes=50)
            results.append(mean_reward)
        
        print(f"Torso Mass: {torso_mass}, Mean of seeds mean rewards: {np.mean(results)}, Max of seeds mean rewards: {results[np.argmax(results)]}")

        # save the model with the highest performance
        model_to_save = models[np.argmax(results)]
        
        # Save the trained model and environment statistics
        model_to_save.save(f'models/hopper_torso_mass_{torso_mass}.zip')
        env.save(f'env/hopper_v4_vecnormalize_torso_mass_{torso_mass}.pkl')
        

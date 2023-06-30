import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
import pickle

# def train_expert(env_id, best_params):
def train_expert(env_id):
    print("Training an expert.")
    env = make_vec_env(env_id, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        verbose=0,
        batch_size=256,
        gamma=0.99,
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
    expert.learn(total_timesteps=900000)
    
    # Evaluate the trained policy
    reward, _ = evaluate_policy(
        expert.policy,  # type: ignore[arg-type]
        expert.env,
        n_eval_episodes=2,
        render=False,
    )
    print(f"Reward after training: {reward}")
    
    return expert
    
if __name__ == '__main__':
    env_id = 'Ant-v4'
    
    # for Optuna hyperparameter optimization
    # with open('env/best_params_task2.pkl', 'rb') as f:
    #     best_params = pickle.load(f)

    # # Now you can access the loaded best_params dictionary
    # print(f"Best params: {best_params}")
    
    # Create the Gym environment
    env = gym.make(env_id, render_mode='human')

    # expert = train_expert(env_id, best_params) # for Optuna hyperparameter optimization
    expert = train_expert(env_id)
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = expert.predict(obs[0])
        env.step(action)
        env.render()
    
    env.close()
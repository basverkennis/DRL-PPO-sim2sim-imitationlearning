import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import pickle

# for Optuna hyperparameter optimization
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.1, log=True)
    n_epochs = trial.suggest_int('n_epochs', 20, 400)
    ent_coef = trial.suggest_float('ent_coef', 0.00001, 0.001)
    n_steps = trial.suggest_int('n_steps', 100, 250)
    batch_size = n_steps * 4
    gamma = trial.suggest_float('gamma', 0.3, 0.7)
    gae_lambda = trial.suggest_float('gae_lambda', 0.3, 0.7)
    clip_range = trial.suggest_float('clip_range', 0.001, 0.1)
    vf_coef = trial.suggest_float('vf_coef', 0.01, 0.5)

    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        verbose=0,
        batch_size=batch_size,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
    )

    expert.learn(total_timesteps=1000000)  # Train for 100000 environment steps
    
    # Return the mean episode reward as the optimization target
    return evaluate_policy(expert, env, n_eval_episodes=3)[0]

def train_expert(env):
    print("Training an expert.")

    # for Optuna hyperparameter optimization
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=500)

    # # best_params = study.best_params
    # print(f"Best params: {best_params}")
    
    # # Save the best_params dictionary to a file
    # with open('env/best_params_task1.pkl', 'wb') as f:
    #     pickle.dump(best_params, f)

    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        verbose=0,
        batch_size=256,
        gamma=0.99,
        ## for Optuna hyperparameter optimization
        # batch_size=best_params['n_steps']*4, # n_steps * n_envs
        # ent_coef=best_params['ent_coef'],
        # learning_rate=best_params['learning_rate'],
        # n_epochs=best_params['n_epochs'],
        # n_steps=best_params['n_steps'],
        # gae_lambda=best_params['gae_lambda'],
        # clip_range=best_params['clip_range'],
        # vf_coef=best_params['vf_coef'],
    )
    
    expert.learn(total_timesteps=5000000)  # Train with the best hyperparameters

    # Evaluate the trained policy
    reward, _ = evaluate_policy(
        expert.policy,  # type: ignore[arg-type]
        expert.env,
        n_eval_episodes=2,
        render=False,
    )
    print(f"Reward after training: {reward}")
    
    # Save the trained model and environment statistics
    expert.save('models/hopper_base.zip')
    env.save('env/hopper_v4_vecnormalize_base.pkl')

    return expert

if __name__ == "__main__":
    # Create the vectorized environment with custom torso masses
    n_envs = 4  # Number of parallel environments
    seed = 0  # Seed for reproducibility
    
    env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    expert = train_expert(env)

# System libraries
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust the path as needed

# Helper function custom library
import helper_functions.qubit_mover_2 as qm2
import helper_functions.enviornments as envs 


# numerical libraries
import sympy as sp
import numpy as np
import datetime

# Visual libraries
import numpy.typing 
import matplotlib.pyplot as plt
import json

# RL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from stable_baselines3.common.callbacks import BaseCallback
import optuna

#from stable_baselines3.common.vec_env import VecNormaliz

# -------------------------------------------------------
#Create logging functions
# -------------------------------------------------------
def save_experiment_results(hyperparameters, callback_params, final_metrics, filename=None):
    """
    Save hyperparameters and results to a text file.
    
    Args:
        hyperparameters: Dict of model hyperparameters
        callback_params: Dict of callback parameters
        final_metrics: Dict of final performance metrics
        filename: Optional custom filename
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/machine_learning_logs/experiment_results_{timestamp}.txt"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("QUANTUM NETWORK PROTOCOL OPTIMIZATION EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Environment: QubitMoverEnv_2_Mapped\n")
        f.write(f"Algorithm: Proximal Policy Optimization (PPO)\n\n")
        
        # Model Hyperparameters
        f.write("MODEL HYPERPARAMETERS:\n")
        f.write("-" * 40 + "\n")
        for key, value in hyperparameters.items():
            f.write(f"{key:25}: {value}\n")
        
        # Callback Parameters
        f.write("\nCALLBACK PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        for key, value in callback_params.items():
            f.write(f"{key:25}: {value}\n")
        
        # Final Results
        f.write("\nFINAL PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                f.write(f"{key:25}: {value:.6f}\n")
            else:
                f.write(f"{key:25}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")

def save_hyperparams(trial, score, folder="data/machine_learning_logs/hyperparameter_logs"):
    os.makedirs(folder, exist_ok=True)
    params = trial.params
    params["score"] = score
    with open(f"{folder}/trial_{trial.number}_params.json", "w") as f:
        json.dump(params, f, indent=4)

# -------------------------------------------------------
# Create evaluation function
# -------------------------------------------------------

def evaluate_model(model, env=None, n_eval_episodes=100, deterministic=True):
    """
    Evaluate a trained model and return the mean reward.
    
    Args:
        model: Trained PPO model
        env: Environment to evaluate on (if None, creates new environment)
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions
        
    Returns:
        float: Mean reward across evaluation episodes
    """
    # Create environment if not provided
    if env is None:
        eval_env = envs.QubitMoverEnv_2_Mapped()
    else:
        eval_env = env
    
    episode_rewards = []
    
    for episode in range(n_eval_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            # Handle truncation (if environment has time limits)
            if truncated:
                done = True
        
        episode_rewards.append(episode_reward)
        
        # Optional: Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            current_mean = np.mean(episode_rewards)
            print(f"Evaluation progress: {episode + 1}/{n_eval_episodes} episodes, current mean: {current_mean:.4f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"Evaluation complete: Mean reward = {mean_reward:.4f} ± {std_reward:.4f} over {n_eval_episodes} episodes")
    
    return mean_reward


# -------------------------------------------------------
# Create custom callback for early function
# -------------------------------------------------------
class RollingAveragePlateauCallback(BaseCallback):
    """
    Stop when the rolling average of rewards plateaus.
    """
    def __init__(self, 
                 window_size: int = 20, # Window to average over
                 patience: int = 15, # How many checks to wait
                 min_delta: float = 0.005, # How much for an improvement
                 check_freq: int = 2000, # How often to check
                 verbose: int = 0): # Level of print messages
        
        super().__init__(verbose)
        self.window_size = window_size
        self.patience = patience
        self.min_delta = min_delta
        self.check_freq = check_freq
        self.reward_buffer = []
        self.best_avg = -np.inf
        self.wait = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0: # Every 2000 steps
            if len(self.model.ep_info_buffer) > 0: # gets the latest reward
                current_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                self.reward_buffer.append(current_reward)
                
                # Keep only the last window_size rewards
                if len(self.reward_buffer) > self.window_size:
                    self.reward_buffer.pop(0)
                
                # Calculate rolling average
                if len(self.reward_buffer) >= self.window_size:
                    current_avg = np.mean(self.reward_buffer)
                    
                    if current_avg > self.best_avg + self.min_delta:
                        self.best_avg = current_avg
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"Step {self.num_timesteps}: New best rolling avg: {current_avg:.4f}")
                    else:
                        self.wait += 1
                        if self.verbose > 0:
                            print(f"Step {self.num_timesteps}: Plateau check {self.wait}/{self.patience}. Avg: {current_avg:.4f}")

                    if self.wait >= self.patience:
                        if self.verbose > 0:
                            print(f"Training stopped due to plateau in rolling average.")
                            print(f"Best rolling average: {self.best_avg:.4f}")
                        return False
                    
    

        return True
    

    def get_final_metrics(self):
        """Get final performance metrics for logging."""
        if len(self.model.ep_info_buffer) > 0:
            self.total_episodes = len(self.model.ep_info_buffer)
            self.final_mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        
        return {
            "final_mean_reward": self.final_mean_reward,
            "best_rolling_average": self.best_avg,
            "total_episodes": self.total_episodes,
            "final_timesteps": self.num_timesteps,
            "patience_wait_count": self.wait
        }

# -------------------------------------------------------
# Hyperparameter optimization with Optuna
# -------------------------------------------------------

def objective(trial):
    

    # Define hyperparameters dictionary
    hyperparameters = {
        "policy": "MlpPolicy",

        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3), 
        # Determines how quickly the model adapts, smaller learning rate = smoother learning

        "n_steps": trial.suggest_categorical("n_steps", [2048, 4096, 8192]), 
        # Number of steps to run for each environment per update, larger = more stable updates

        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]), 
        # Minibatch size for each gradient update, Larger = more stable updates

        "n_epochs": trial.suggest_int("n_epochs", 10, 30), 
        # Number of epochs to perform for each update, larger = more learning per update

        "gamma": trial.suggest_uniform("gamma", 0.95, 0.999), 
        # Discount factor, close to 1 means future rewards are considered more

        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.95, 0.99), 

        "clip_range":  trial.suggest_uniform("clip_range", 0.1, 0.3),

        "clip_range_vf": None,

        "normalize_advantage": True,

        "ent_coef": trial.suggest_uniform("ent_coef", 0.01, 0.1),

        "vf_coef": trial.suggest_uniform("vf_coef", 0.25, 0.75),

        "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.3, 0.7),

        "use_sde": False,

        "sde_sample_freq": -1,

        "target_kl": None,

        "stats_window_size": 100,

        "total_timesteps": 200_000 # Total training timesteps lowered for faster sweeps
    }

    # Define callback parameters
    callback_params = {
        "window_size": 20,
        "patience": 20,
        "min_delta": 0.005,
        "check_freq": 2000,
        "callback_type": "RollingAveragePlateauCallback"
    }


    # Create the callback
    plateau_callback = RollingAveragePlateauCallback(
        window_size=callback_params["window_size"],
        patience=callback_params["patience"],
        min_delta=callback_params["min_delta"],
        check_freq=callback_params["check_freq"],
        verbose=1
    )


    # Load in the custom environment
    env=envs.QubitMoverEnv_2_Mapped() # Mapped version


    # Create the RL model
    model = PPO(
        policy=hyperparameters["policy"],
        env=env,
        learning_rate=hyperparameters["learning_rate"],
        n_steps=hyperparameters["n_steps"],
        batch_size=hyperparameters["batch_size"],
        n_epochs=hyperparameters["n_epochs"],
        gamma=hyperparameters["gamma"],
        gae_lambda=hyperparameters["gae_lambda"],
        clip_range=hyperparameters["clip_range"],
        clip_range_vf=hyperparameters["clip_range_vf"],
        normalize_advantage=hyperparameters["normalize_advantage"],
        ent_coef=hyperparameters["ent_coef"],
        vf_coef=hyperparameters["vf_coef"],
        max_grad_norm=hyperparameters["max_grad_norm"],
        use_sde=hyperparameters["use_sde"],
        sde_sample_freq=hyperparameters["sde_sample_freq"],
        target_kl=hyperparameters["target_kl"],
        stats_window_size=hyperparameters["stats_window_size"],
        verbose=0,
        tensorboard_log="data/machine_learning_logs/ppo_qubit_mover2_tensorboard/"
    )


    # Train the model
    model.learn(total_timesteps=hyperparameters["total_timesteps"], callback=plateau_callback)


    # Save the model -> only the best models get saved
    mean_reward = evaluate_model(model,env,n_eval_episodes=100,deterministic=False)

        
    print(f"Trial {trial.number} completed: {mean_reward:.4f}")
    return mean_reward








print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)# just two trials for testing

print("Optimization complete!")
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value:.4f}")
print("Best params:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Retrain the best model and save it
print("\nRetraining best model...")
best_params = study.best_params

# Recreate best hyperparameters
best_hyperparameters = {
    "policy": "MlpPolicy",
    "learning_rate": best_params["learning_rate"],
    "n_steps": best_params["n_steps"],
    "batch_size": best_params["batch_size"],
    "n_epochs": best_params["n_epochs"],
    "gamma": best_params["gamma"],
    "gae_lambda": best_params["gae_lambda"],
    "clip_range": best_params["clip_range"],
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": best_params["ent_coef"],
    "vf_coef": best_params["vf_coef"],
    "max_grad_norm": best_params["max_grad_norm"],
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "stats_window_size": 100,
    "total_timesteps": 1_000_000  # Train longer for final model
}

# Train final best model
env = envs.QubitMoverEnv_2_Mapped()
best_model = PPO(
    policy=best_hyperparameters["policy"],
    env=env,
    learning_rate=best_hyperparameters["learning_rate"],
    n_steps=best_hyperparameters["n_steps"],
    batch_size=best_hyperparameters["batch_size"],
    n_epochs=best_hyperparameters["n_epochs"],
    gamma=best_hyperparameters["gamma"],
    gae_lambda=best_hyperparameters["gae_lambda"],
    clip_range=best_hyperparameters["clip_range"],
    ent_coef=best_hyperparameters["ent_coef"],
    vf_coef=best_hyperparameters["vf_coef"],
    max_grad_norm=best_hyperparameters["max_grad_norm"],
    verbose=1,
    tensorboard_log="data/machine_learning_logs/ppo_qubit_mover2_tensorboard/"
)

plateau_callback = RollingAveragePlateauCallback(
    window_size=20,
    patience=20,
    min_delta=0.005,
    check_freq=2000,
    verbose=1
)

best_model.learn(total_timesteps=best_hyperparameters["total_timesteps"], callback=plateau_callback)

# Save the final best model
final_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
final_model_path = f"data/machine_learning_agents/agents_qubit_mover2/ppo_qubit_mover2_agent_BEST_FINAL_{final_timestamp}"
best_model.save(final_model_path)

# Final evaluation
final_mean_reward = evaluate_model(best_model, env, n_eval_episodes=100, deterministic=True)

print(f"Final best model saved: {final_model_path}")
print(f"Final evaluation score: {final_mean_reward:.4f}")

# Save final experiment results
callback_params = {
    "window_size": 20,
    "patience": 20,
    "min_delta": 0.005,
    "check_freq": 2000,
    "callback_type": "RollingAveragePlateauCallback"
}

final_metrics = {
    "final_mean_reward": final_mean_reward,
    "best_trial_number": study.best_trial.number,
    "optimization_trials": len(study.trials),
    "model_saved_path": final_model_path
}

save_experiment_results(best_hyperparameters, callback_params, final_metrics)





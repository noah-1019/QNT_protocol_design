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

# Visual libraries
import numpy.typing 
import matplotlib.pyplot as plt

# RL libraries
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


# Load in the custom environment
env=envs.QubitMoverEnv_2()

# Create the RL model
model = PPO(
    "MlpPolicy",      # Use a multilayer perceptron policy
    env,              # Pass your custom environment
    verbose=1,        # Print training progress
    tensorboard_log="data/machine_learning_logs/ppo_qubit_mover2_tensorboard/"
)

# Train the model
model.learn(total_timesteps=100_000)

model.save("data/machine_learning_agents/agents_qubit_mover2/ppo_qubit_mover2_agent")

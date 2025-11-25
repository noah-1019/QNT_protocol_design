# Standard libraries
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt


current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)
import qubit_mover_2 as qm2
import qubit_mover_1 as qm1


# ----------------------------------------------------------------------------------
# Qubit Mover Environment for 3 Qubits Invalid actions are not mapped and are penalized severely
# ----------------------------------------------------------------------------------

class QubitMoverEnv_2(gym.Env):
    """
    Custom Environment for the Qubit Mover RL agent.
    - 3 qubits, each with a path of length 10
    - Each step, the agent selects a node (1-3) or 0 (measure) for each qubit as well as whether to apply a hadamard gate (4)
    - Episode ends after a fixed number of steps or when all qubits are measured
    """
    

    def __init__(self):
        super().__init__()
        self.n_qubits = 3
        self.max_steps = 10
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist() # parameter values, used to represent channel failure rates.

        # Each qubit can be at node 0, 1, 2, ... , 13 at each step see helper_functions/qubit_mover_2.py  for details
        ## Encoding Scheme:
        # 1-4: Node 1 with H=0,1,2,3
        # 5-8: Node 2 with H=0,1,2,3
        # 9-12: Node 3 with H=0,1,2,3
        # 13: Measurement (end move)

        ## Returned Values:
        # A list of integers representing the path taken, where:
        # 0: No Hadamard gate applied
        # 4: Hadamard gate applied
        # Error Operators are represented by their number (1, 2, or 3)

        self.action_space = spaces.MultiDiscrete([13] * self.n_qubits) # 3 qubits, each with 13 possible actions Which will be mapped to 1-13
        # Actions: 0-12, which will be mapped to 1-13


        # Observation space: qubit states + theta values
        # The Agent can observe the full path taken so far (n_qubits * max_steps) +  the 3 parameter values.
        low = np.zeros(self.n_qubits * self.max_steps + 3)
        high = np.concatenate([np.full(self.n_qubits * self.max_steps, 14), np.ones(3)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        """reset
        Resets the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)# Reset the seed if provided
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist() # parameter values, used to represent channel failure rates.
        # Each channel has a random failure rate between 0.05 and 0.5
         
        self.state = np.zeros((self.n_qubits, self.max_steps), dtype=np.int32) # Initialize state to zeros
        self.state=self.state.flatten() # The moves that have been taken so far.
        self.current_step = 0 # Current step in the episode
        self.done = False # Whether the episode is done
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)
        return obs, {}
    
    def step(self, action):
        # action: array of length 3, each in {0,1,2,3}
        mapped_action = [a + 1 for a in action]  # Convert 0-12 to 1-13

        if self.done:
            raise RuntimeError("Episode is done")

        # Record the action for each qubit at this step
        # Update the observation space
        for q in range(self.n_qubits): # Iterate through each qubit
            self.state[q * self.max_steps + self.current_step] = mapped_action[q] # Update the state with the action taken

        self.current_step += 1 # Move to the next step

        # If all qubits have been measured (0) or max_steps reached, episode is done
        state_2d = self.state.reshape(self.n_qubits, self.max_steps)
        all_measured = np.all(state_2d[:, self.current_step-1] == 13)

        self.done = all_measured or self.current_step >= self.max_steps

        reward_val = 0 # Default reward
        
        # The reward is only calculated at the end of the episode
        

        if self.done:
            # Convert state to list of lists for your reward function
            node_lists = [list(state_2d[q]) for q in range(self.n_qubits)]
            reward_val = qm2.reward_direct(node_lists, self.thetas) # Calculate reward using the direct reward function
        else:
            reward_val = 0  # Or a step penalty if desired

        self.state=self.state.flatten() # gymnasium requires 1D array for Box space
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)

        return obs.copy(), reward_val, self.done, False, {}

    def render(self):
        print("Step:", self.current_step)
        print("State:\n", self.state)

def test_QubitMoverEnv_2():
    env = QubitMoverEnv_2()
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")  # Should be (33,)
    print(f"Action space: {env.action_space}")  # MultiDiscrete([13 13 13])
    
    # Test valid actions
    action = [0, 4, 8]  # Maps to [1, 5, 9] - valid starting positions
    obs, reward, done, truncated, info = env.step(action)
    print(f"First step - Reward: {reward}, Done: {done}")
    
    # Test measurement action
    action = [12, 12, 12]  # Maps to [13, 13, 13] - measurement
    obs, reward, done, truncated, info = env.step(action)
    print(f"Measurement step - Reward: {reward}, Done: {done}")

    check_env(env)  # This will raise an error if the environment is not valid

def visualize_QubitMoverEnv_2(iterations: int=100):
    qm2.plot_random_rewards_histogram(iterations)


# ----------------------------------------------------------------------------------
# Qubit Mover Environment Mapped Invalid actions are mapped to the nearest valid action
# -----------------------------------------------------------------------------------
class QubitMoverEnv_2_Mapped(gym.Env):
    """
    Custom Environment for the Qubit Mover RL agent.
    - 3 qubits, each with a path of length 10
    - Each step, the agent selects a node (1-3) or 0 (measure) for each qubit as well as whether to apply a hadamard gate (4)
    - Episode ends after a fixed number of steps or when all qubits are measured
    """
    

    def __init__(self):
        super().__init__()
        self.n_qubits = 3
        self.max_steps = 10
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist() # parameter values, used to represent channel failure rates.

        # Each qubit can be at node 0, 1, 2, ... , 13 at each step see helper_functions/qubit_mover_2.py  for details
        ## Encoding Scheme:
        # 1-4: Node 1 with H=0,1,2,3
        # 5-8: Node 2 with H=0,1,2,3
        # 9-12: Node 3 with H=0,1,2,3
        # 13: Measurement (end move)

        ## Returned Values:
        # A list of integers representing the path taken, where:
        # 0: No Hadamard gate applied
        # 4: Hadamard gate applied
        # Error Operators are represented by their number (1, 2, or 3)

        self.action_space = spaces.MultiDiscrete([13] * self.n_qubits) # 3 qubits, each with 13 possible actions Which will be mapped to 1-13
        # Actions: 0-12, which will be mapped to 1-13


        # Observation space: qubit states + theta values
        # The Agent can observe the full path taken so far (n_qubits * max_steps) +  the 3 parameter values.
        low = np.zeros(self.n_qubits * self.max_steps + 3)
        high = np.concatenate([np.full(self.n_qubits * self.max_steps, 14), np.ones(3)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        """reset
        Resets the environment to an initial state and returns an initial observation.
        """
        super().reset(seed=seed)# Reset the seed if provided
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist() # parameter values, used to represent channel failure rates.
        # Each channel has a random failure rate between 0.05 and 0.5
         
        self.state = np.zeros((self.n_qubits, self.max_steps), dtype=np.int32) # Initialize state to zeros
        self.state=self.state.flatten() # The moves that have been taken so far.
        self.current_step = 0 # Current step in the episode
        self.done = False # Whether the episode is done
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)
        return obs, {}
    
    def step(self, action):
        # action: array of length 3, each in {0,1,2,3}
        mapped_action = [a + 1 for a in action]  # Convert 0-12 to 1-13

        if self.done:
            raise RuntimeError("Episode is done")

        # Record the action for each qubit at this step
        # Update the observation space
        for q in range(self.n_qubits): # Iterate through each qubit
            self.state[q * self.max_steps + self.current_step] = mapped_action[q] # Update the state with the action taken

        self.current_step += 1 # Move to the next step

        # If all qubits have been measured (0) or max_steps reached, episode is done
        state_2d = self.state.reshape(self.n_qubits, self.max_steps)
        all_measured = np.all(np.any(state_2d[:, :self.current_step] == 13, axis=1))
        self.done = all_measured or self.current_step >= self.max_steps

        reward_val = 0 # Default reward
        
        # The reward is only calculated at the end of the episode
        

        if self.done:
            # Convert state to list of lists for your reward function
            node_lists = [list(state_2d[q]) for q in range(self.n_qubits)]
            reward_val = qm2.reward_direct(node_lists, 
                                           self.thetas,
                                           mapped=True,
                                           mapped_penalty=0.03) # Calculate reward using the direct reward function
        else:
            reward_val = 0  # Or a step penalty if desired

        self.state=self.state.flatten() # gymnasium requires 1D array for Box space
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)

        return obs.copy(), reward_val, self.done, False, {}

    def render(self):
        print("Step:", self.current_step)
        print("State:\n", self.state)


def test_QubitMoverEnv_2_Mapped():
    env = QubitMoverEnv_2_Mapped()
    obs, info = env.reset()
    
    print(f"Observation shape: {obs.shape}")  # Should be (33,)
    print(f"Action space: {env.action_space}")  # MultiDiscrete([13 13 13])
    
    # Test valid actions
    action = [0, 4, 8]  # Maps to [1, 5, 9] - valid starting positions
    obs, reward, done, truncated, info = env.step(action)
    print(f"First step - Reward: {reward}, Done: {done}")
    
    # Test measurement action
    action = [12, 12, 12]  # Maps to [13, 13, 13] - measurement
    obs, reward, done, truncated, info = env.step(action)
    print(f"Measurement step - Reward: {reward}, Done: {done}")

    check_env(env)  # This will raise an error if the environment is not valid

def visualize_QubitMoverEnv_2_Mapped(iterations: int=100):
    rewards=[]
    for i in range(iterations):
        move=[np.random.randint(1,13,size=np.random.randint(1,10)).tolist() for i in range(3)]
        temp_reward=qm2.reward_direct(move, np.random.uniform(0.05, 0.5, 3).tolist(),mapped=True,mapped_penalty=0.03)
        rewards.append(temp_reward)

    print(f"Average reward over {iterations} random moves: {np.mean(rewards)}")
    print(f"Max reward over {iterations} random moves: {np.max(rewards)}")
    print(f"Min reward over {iterations} random moves: {np.min(rewards)}")

    # Simple histogram
    plt.hist(rewards, bins=30)
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.show()


class QubitMoverEnv1(gym.Env):
    """
    Custom Environment for the Qubit Mover RL agent.
    - 3 qubits, each with a path of length 10
    - Each step, the agent selects a node (1-3) or 0 (measure) for each qubit
    - Episode ends after a fixed number of steps or when all qubits are measured
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        self.n_qubits = 3
        self.max_steps = 10
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist()

        # Each qubit can be at node 0, 1, 2, or 3 at each step
        self.action_space = spaces.MultiDiscrete([4] * self.n_qubits)


        # Observation space: qubit states + theta values
        # Qubit states: (n_qubits * max_steps), Theta: (3,)
        low = np.zeros(self.n_qubits * self.max_steps + 3)
        high = np.concatenate([np.full(self.n_qubits * self.max_steps, 3), np.ones(3)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.thetas = np.random.uniform(0.05, 0.5, 3).tolist() 
        self.state = np.zeros((self.n_qubits, self.max_steps), dtype=np.int32)
        self.state=self.state.flatten()
        self.current_step = 0
        self.done = False
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)
        return obs, {}


    def step(self, action):
        # action: array of length 3, each in {0,1,2,3}
        if self.done:
            raise RuntimeError("Episode is done")

        # Record the action for each qubit at this step
        # Update the observation space
        for q in range(self.n_qubits):
            self.state[q * self.max_steps + self.current_step] = action[q]

        self.current_step += 1

        # If all qubits have been measured (0) or max_steps reached, episode is done
        state_2d = self.state.reshape(self.n_qubits, self.max_steps)
        all_measured = np.all(state_2d[:, self.current_step-1] == 0)

        self.done = all_measured or self.current_step >= self.max_steps

        reward_val = 0
        
        # The reward is only calculated at the end of the episode
        if self.done:
            # Convert state to list of lists for your reward function
            node_lists = [list(state_2d[q]) for q in range(self.n_qubits)]
            reward_val = qm1.reward(node_lists, self.thetas)
        else:
            reward_val = 0  # Or a step penalty if desired

        self.state=self.state.flatten()
        obs = np.concatenate([self.state, np.array(self.thetas, dtype=np.float32)]).astype(np.float32)

        return obs.copy(), reward_val, self.done, False, {}

    def render(self):
        print("Step:", self.current_step)
        print("State:\n", self.state)

    def close(self):
        pass
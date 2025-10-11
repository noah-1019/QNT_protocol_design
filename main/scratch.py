import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust the path as needed
import helper_functions.qubit_mover_2 as qm2
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.typing 


def choose_random_valid_move(params=[0.1,0.2,0.3]):
    moves = [[], [], []]  # Three qubits
    nodes=[1, 5, 9]  # Starting nodes for the three qubits
    for i in range(3): # Add this loop to generate moves for each qubit
        num_moves = np.random.randint(3, 5)  # Between 3 and 10 moves for each qubit
        moves[i]= [random.choice(nodes) for _ in range(num_moves)]

        # Add Hadamard gates
        h_counter=0 # Counts the number of Hadamard gates added
        for j in range(len(moves[i])-1):
            hadamard_orientation=np.random.randint(0,4) # Randomly choose one of the 4 Hadamard orientations

            if hadamard_orientation == 1 or hadamard_orientation == 2:
                h_counter+=1
            elif hadamard_orientation == 3:
                h_counter+=2


            moves[i][j]+=hadamard_orientation # Append the orientation to the node number

        if h_counter % 2 == 1: # If there's an odd number of Hadamard gates, add one more at the end
            moves[i].append(random.choice([2,6,10])) # Append a Hadamard gate at the end, each one of these moves is a hadamard gate plus a move.
            h_counter+=1

        # Ensure the last move is always a measurement
        moves[i].append(13) # Finally, append the measurement operation

        # paths=qm2.nodes_to_paths(moves[i])
        # print("*"*50)
        # print("Moves:", moves[i])
        # print("Num Hadamardds:", h_counter)
        # print("Paths: ", paths )
        # print("Other H calculation: ",paths.count(4))
        

    reward=qm2.reward_numeric(moves, thetas=params)

    

    return reward



    


nodes=[1,6,11,9,13]# Simple case
#nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement

paths = qm2.nodes_to_paths(nodes)



rhos = qm2.paths_to_gates_direct(paths, [0.1, 0.2, 0.3])


# Define parameter values for p1, p2, p3


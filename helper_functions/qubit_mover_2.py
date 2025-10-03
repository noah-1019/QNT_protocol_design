"""
Qubit mover 2 helper functions. This is a file containg helper functions for the qubit mover 2 reinforcement learning agent.

- Author: Noah Plant
- Date: September 2025

Functions:
- error_1, error_2, error_3: Apply quantum error models
- hadamard: Apply Hadamard gate transformation

"""

### Import necessary libraries
# QFIM calculations libraries
from itertools import combinations

import sympy as sp
# Estimator libraries
from scipy.optimize import root, least_squares

# Standard libraries
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt

# ============================================================================
# GLOBAL QUANTUM CONSTANTS - Initialize once, use everywhere
# ============================================================================

# Quantum states
STATE0 = sp.Matrix([[1, 0], [0, 0]])

# Quantum gates
X_GATE = sp.Matrix([[0, 1], [1, 0]])
H_GATE = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)

# Parameter symbols - true parameters
p1, p2, p3 = sp.symbols('p1 p2 p3', real=True)
SYMBOLS_LIST = [p1, p2, p3]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_symbols():
    """Get all parameter symbols used in the quantum operations.
    
    Returns:
        tuple: (true_params, estimated_params)
    """
    return SYMBOLS_LIST.copy()

def get_gates():
    """Get all quantum gate matrices.
    
    Returns:
        dict: Dictionary containing all gate matrices
    """
    return {
        'X': X_GATE.copy(),
        'H': H_GATE.copy(),
        'state0': STATE0.copy()
    }

# Utility functions to access gates

def get_x_gate():
    """Return a copy of the Pauli-X gpate as a sympy Matrix."""
    return X_GATE.copy()

def get_h_gate():
    """Return a copy of the Hadamard gate as a sympy Matrix."""
    return H_GATE.copy()

def get_state0():
    """Return a copy of the |0><0| state as a sympy Matrix."""
    return STATE0.copy()

# ============================================================================
# QUANTUM ERROR FUNCTIONS - Now clean and efficient
# ============================================================================

def error_1(rho):
    """Apply error model 1 to density matrix rho.
    
    Args:
        rho: Density matrix (sympy Matrix)
        
    Returns:
        Modified density matrix with error_1 applied
    """
    return (1-p1) * rho + p1 * X_GATE * rho * X_GATE

def error_2(rho):
    """Apply error model 2 to density matrix rho.
    
    Args:
        rho: Density matrix (sympy Matrix)
        
    Returns:
        Modified density matrix with error_2 applied
    """
    return (1-p2) * rho + p2 * X_GATE * rho * X_GATE

def error_3(rho):
    """Apply error model 3 to density matrix rho.
    
    Args:
        rho: Density matrix (sympy Matrix)
        
    Returns:
        Modified density matrix with error_3 applied
    """
    return (1-p3) * rho + p3 * X_GATE * rho * X_GATE

def hadamard(rho):
    """Apply Hadamard gate to density matrix rho.
    
    Args:
        rho: Density matrix (sympy Matrix)
        
    Returns:
        Transformed density matrix H * rho * H†
    """
    return H_GATE * rho * H_GATE

def calculate_QFIM(rhos,thetas=None):
    # compute eigen values

    F=sp.zeros(len(SYMBOLS_LIST), len(SYMBOLS_LIST))
    n=len(SYMBOLS_LIST)

    for rho in rhos:

        lambdas=rho.eigenvals()


        for i in range(n):
            for j in range(n):
                s = 0
                for lam in lambdas:
                    if lam == 0:
                        continue  # Skip zero eigenvalues to avoid division by zero
                    s += (1/lam) * lam.diff(SYMBOLS_LIST[i]) * lam.diff(SYMBOLS_LIST[j])
                F[i, j] += s


    if thetas is not None:
        F=F.subs({p1: thetas[0], p2: thetas[1], p3: thetas[2]})
    return F

# ============================================================================
# Move to Path Conversion
# ============================================================================
def remove_padding_13s(move_list):
    """Remove all 13's except the first one encountered."""
    result = []
    found_first_13 = False
    
    for move in move_list:
        if move == 13:
            if not found_first_13:
                result.append(move)
                found_first_13 = True
            # Skip all subsequent 13's
        else:
            result.append(move)
    
    return result

def nodes_to_paths(move_list: list[int]):
    # Each element in the move list is an integer from 1 to 13

    ## Encoding Scheme:
    # 1-4: Node 1 with H=0,1,2,3
    # 5-8: Node 2 with H=0,1,2,
    # 9-12: Node 3 with H=0,1,2,3
    # 13: Measurement (end move)

    ## Returned Values:
    # A list of integers representing the path taken, where:
    # 0: No Hadamard gate applied
    # 4: Hadamard gate applied
    # Error Operators are represented by their number (1, 2, or 3)


    # Remove padding 13's except the first one
    move_list = remove_padding_13s(move_list)

    #---------------------------------------------------------------------------
    # Verify correct input
    # ---------------------------------------------------------------------------
    num_moves=len([n for n in move_list if n!=13]) # Counts the number of non-zero moves
    if num_moves<2:
        # Must have at least a start and end move
        return -1 # Invalid single node list
    
    if move_list[-1]!=13:
        # Must end on a measurement
        return -1 # Invalid if last move is not a 13 (end move)
    
    if any(move == 13 for move in move_list[1:-1]):  
        # Cannot measure in the middle of the distribution protocol
        return -1 # Invalid if any move other than last is a 13 (end move)
    
    move_list.pop() # Remove the measurement for processing

    #---------------------------------------------------------------------------
    # Extract nodes and hadamard gates from move list
    # ---------------------------------------------------------------------------

    node_list=[((n-1)//4)+1 for n in move_list] 
    h_list=[((n-1)%4) for n in move_list] # Extracts the hadamard gates from the list
    #print(h_list)
    # print("moves: " ,move_list)
    # print("nodes: " ,node_list)
    # print("Hads:  " ,h_list)

    #---------------------------------------------------------------------------
    # Convert to path list
    # ---------------------------------------------------------------------------

    path_list=[]
    for i in range(len(node_list)):
        if h_list[i]==0:
            path_list.append(node_list[i])
            path_list.append(0)
            path_list.append(node_list[i])
            path_list.append(0)

        elif h_list[i]==1:
            path_list.append(node_list[i])
            path_list.append(4)
            path_list.append(node_list[i])
            path_list.append(0)
            

        elif h_list[i]==2:
            path_list.append(node_list[i])
            path_list.append(0)
            path_list.append(node_list[i])
            path_list.append(4)
        elif h_list[i]==3:
            
            path_list.append(node_list[i])
            path_list.append(4)
            path_list.append(node_list[i])
            path_list.append(4)

    
    filtered_path_list=path_list[1:-2] # Remove the initial and final padding zeros

    return filtered_path_list


def moves_to_gates(move_list):
    temp_rho = get_state0()  # Could use global directly since operations are safe
    
    for move in move_list:
        if move == 1:
            temp_rho = error_1(temp_rho)  # Use existing functions
        elif move == 2:
            temp_rho = error_2(temp_rho)
        elif move == 3:
            temp_rho = error_3(temp_rho)
        elif move == 4:
            temp_rho = hadamard(temp_rho)
        elif move == 0:
            continue
        else:
            return -10
    
    return temp_rho
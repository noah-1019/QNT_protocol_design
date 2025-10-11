import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import sympy as sp
import numpy as np
from helper_functions import qubit_mover_2 as qm2

# ============================================================================
# Test helper functions in qubit_mover_2.py
# ============================================================================


# ============================================================================
# Test get functions and custom quantum operations
# ============================================================================

def test_error_1_identity():
    # Test error_1 with identity input
    rho = sp.Matrix([[1, 0], [0, 0]])
    result = qm2.error_1(rho)
    # Should be a linear combination of rho and X*rho*X
    X = qm2.get_x_gate()
    expected = (1-qm2.p1) * rho + qm2.p1 * X * rho * X
    assert sp.simplify(result - expected) == sp.zeros(2)

def test_error_2_identity():
    rho = sp.Matrix([[1, 0], [0, 0]])
    result = qm2.error_2(rho)
    X = qm2.get_x_gate()
    expected = (1-qm2.p2) * rho + qm2.p2 * X * rho * X
    assert sp.simplify(result - expected) == sp.zeros(2)

def test_error_3_identity():
    rho = sp.Matrix([[1, 0], [0, 0]])
    result = qm2.error_3(rho)
    X = qm2.get_x_gate()
    expected = (1-qm2.p3) * rho + qm2.p3 * X * rho * X
    assert sp.simplify(result - expected) == sp.zeros(2)

def test_hadamard_on_state0():
    rho = sp.Matrix([[1, 0], [0, 0]])
    H = qm2.get_h_gate()
    result = qm2.hadamard(rho)
    expected = H * rho * H
    assert sp.simplify(result - expected) == sp.zeros(2)

def test_get_gates():
    gates = qm2.get_gates()
    assert 'X' in gates and 'H' in gates
    assert isinstance(gates['X'], sp.Matrix)
    assert gates['X'].shape == (2, 2)

def test_get_state0():
    state0 = qm2.get_state0()
    expected = sp.Matrix([[1, 0], [0, 0]])
    assert state0 == expected

# ============================================================================
# Calculate QFIM
# ============================================================================

def test_calculate_QFIM():
    rho=[sp.Matrix([[1, 0], [0, 1]])]
    result = qm2.calculate_QFIM(rho)

    expected=sp.Matrix([[0, 0,0], [0, 0,0], [0, 0, 0]])

    assert result.shape == (3, 3)
    assert all(isinstance(entry, sp.Basic) for entry in result)
    assert sp.simplify(result - expected) == sp.zeros(3)
    
def test_calculate_QFIM_theta_dependence():

    symbols = qm2.get_all_symbols()

    rho=[sp.Matrix([[symbols[1],0],[0,symbols[2]] ])]
    result = qm2.calculate_QFIM(rho)

    expected=sp.Matrix([[0, 0,0], [0, 1/symbols[1],0], [0, 0, 1/symbols[2]]])

    assert result.shape == (3, 3)
    assert all(isinstance(entry, sp.Basic) for entry in result)
    assert sp.simplify(result - expected) == sp.zeros(3)

def test_calculate_QFIM_multiple_rho():
    symbols = qm2.get_all_symbols()

    rho1=sp.Matrix([[symbols[1],0],[0,symbols[2]] ])
    rho2=sp.Matrix([[symbols[0],0],[0,1] ])
    result = qm2.calculate_QFIM([rho1,rho2])
    expected=sp.Matrix([[1/symbols[0], 0,0], [0, 1/symbols[1],0], [0, 0, 1/symbols[2]]])

    assert result.shape == (3, 3)
    assert all(isinstance(entry, sp.Basic) for entry in result)
    assert sp.simplify(result - expected) == sp.zeros(3)

def test_calculate_QFIM_edge_case(): # zero eigenvalue
    symbols= qm2.get_all_symbols()

    rho=[sp.Matrix([[symbols[0],0],[0,0] ])] # Edge case with zero eigenvalue
    result = qm2.calculate_QFIM(rho)

    expected=sp.Matrix([[1/symbols[0], 0,0], [0, 0,0], [0, 0, 0]])

    assert result.shape == (3, 3)
    assert all(isinstance(entry, sp.Basic) for entry in result)
    assert sp.simplify(result - expected) == sp.zeros(3)

def test_calculate_QFIM_given_theta():
    symbols = qm2.get_all_symbols()

    rho=[sp.Matrix([[symbols[1],0],[0,symbols[2]] ])]
    result = qm2.calculate_QFIM(rho, thetas=[0.5, 0.3, 0.2])

    expected=sp.Matrix([[0, 0,0], [0, 1/(0.3),0], [0, 0, 1/0.2]])

    assert result.shape == (3, 3)
    assert all(isinstance(entry, sp.Basic) for entry in result)
    assert sp.simplify(result - expected) == sp.zeros(3)

def test_calculate_QCRB():
    F=sp.Matrix([[2,0,0],[0,3,0],[0,0,4]])
    result=qm2.calculate_QCRB(F)
    print(result)
    expected=1/3 + 1/4 + 1/2 # Trace of the inverse
    assert sp.simplify(result - expected) == 0


# ============================================================================
# Test nodes_to_paths function
# ============================================================================

def test_nodes_to_paths_valid():
    moves = [1, 5, 9, 13]
    expected = [0, 1, 0, 2, 0, 2,0,3,0]  # No Hadamard gates applied
    result = qm2.nodes_to_paths(moves)
    assert result == expected

def test_nodes_to_paths_hadamard():
    moves = [4, 8, 12, 13]
    expected = [4, 1, 4, 2, 4, 2,4,3,4]  # Hadamard gates applied at each node
    result = qm2.nodes_to_paths(moves)
    assert result == expected

def test_nodes_to_paths_measurement_in_middle():
    moves = [1, 13, 5, 9]
    result = qm2.nodes_to_paths(moves)
    assert result == -1  # Invalid due to measurement in the middle

def test_nodes_to_paths_no_measurement():
    moves = [1, 5, 9]
    result = qm2.nodes_to_paths(moves)
    assert result == -1  # Invalid due to no measurement at the end

def test_nodes_to_paths_single_node():
    moves = [1, 13]
    result = qm2.nodes_to_paths(moves)
    assert result == -1  # Invalid due to single node

def test_nodes_to_paths_valid_padding():
    moves = [1, 5, 9, 13,13,13,13,13,13]
    expected = [0, 1, 0, 2, 0, 2,0,3,0]  # No Hadamard gates applied
    result = qm2.nodes_to_paths(moves)
    assert result == expected

# ============================================================================
# Test paths_to_gates function
# ============================================================================

def test_paths_to_gates_simple():
    moves=[1,4]
    result=qm2.paths_to_gates(moves)
    result=sp.simplify(result)


    p1, _, _ = qm2.get_all_symbols()

    expected=sp.Matrix([[1/2,1/2-p1],[1/2-p1,1/2]])



    assert result.shape == (2, 2)
    assert sp.simplify(result - expected) == sp.zeros(2, 2)

def test_paths_to_gates_with_0():
    moves=[1,0,4,0,0,0]
    result=qm2.paths_to_gates(moves)
    result=sp.simplify(result)


    p1, _, _ = qm2.get_all_symbols()

    expected=sp.Matrix([[1/2,1/2-p1],[1/2-p1,1/2]])



    assert result.shape == (2, 2)
    assert sp.simplify(result - expected) == sp.zeros(2, 2)

def test_paths_to_gates_moves_list():
    moves=[1,12,4,8,13]
    gates=qm2.nodes_to_paths(moves)

    result=qm2.paths_to_gates(gates)
    #result=sp.simplify(result) # Too complex to simplify easily

    assert result.shape == (2, 2)

# ============================================================================
# Test Reward Calculation
# ============================================================================

def test_normalize_reward():
    qcrb_values = [0.01, 0.1, 1, 10, 100,1000]
    normalized = [qm2.normalize_reward(qcrb) for qcrb in qcrb_values]
    
    # Check that rewards are between 0 and 1
    assert all(0 <= r <= 1 for r in normalized)
    
    # Check that lower QCRB gives higher reward
    assert normalized == sorted(normalized, reverse=True)
    
   
def test_reward_valid():
    moves = [[2, 5, 10, 1, 13], [5, 5, 9, 1, 13], [9, 5, 9, 1, 13]]
    result = qm2.reward(moves,[0.1,0.2,0.3])
    assert isinstance(result, float)
    assert -1 <= result <= 1  # Reward should be normalized between -1 and 1

def test_reward_invalid_move():
    moves = [[1, 13, 5, 9]]  # Measurement in the middle
    result = qm2.reward(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid move should return -1

def test_reward_odd_hadamards():
    moves = [[ 1, 5, 10, 13]]  # Odd number of Hadamards
    result = qm2.reward(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid due to odd Hadamards should return -1


# ============================================================================
# Test Numeric QFIM Calculation
# ============================================================================

def test_calculate_QFIM_numerical():
    nodes=[1,6,11,9,13]# Simple case
    #nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths = qm2.nodes_to_paths(nodes)
    rho = qm2.paths_to_gates(paths)
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_numerical([rho], params,debug=True)
    
    


    qfim2=qm2.calculate_QFIM([rho],thetas=params)
    print(qfim2)

    qfim_numpy=np.array(qfim2.evalf(), dtype=complex)

    assert np.allclose(F, qfim_numpy, atol=1e-5), f"Numeric QFIM {F} does not match symbolic QFIM {qfim_numpy}"
    assert F.shape == (3, 3)
   
def test_calculate_QFIM_numerical_multiple_rho():
    nodes1=[1,6,11,9,13]# Simple case
    nodes2=[4,5,10,1,13]# Simple case with Hadamard
    #nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths1 = qm2.nodes_to_paths(nodes1)
    rho1 = qm2.paths_to_gates(paths1)

    paths2 = qm2.nodes_to_paths(nodes2)
    rho2 = qm2.paths_to_gates(paths2)
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_numerical([rho1,rho2], params,debug=True)
    
    


    qfim2=qm2.calculate_QFIM([rho1,rho2],thetas=params)
    print(qfim2)

    qfim_numpy=np.array(qfim2.evalf(), dtype=complex)

    assert np.allclose(F, qfim_numpy, atol=1e-5), f"Numeric QFIM {F} does not match symbolic QFIM {qfim_numpy}"
    assert F.shape == (3, 3)

def test_calculate_QFIM_numerical_difficult():
    nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths = qm2.nodes_to_paths(nodes)
    rho = qm2.paths_to_gates(paths)
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_numerical([rho], params,debug=True)
    
    assert F.shape == (3, 3)


# ============================================================================
# Test Direct QFIM Calculation
# ============================================================================
def test_calculate_QFIM_direct():
    nodes=[1,6,11,9,13]# Simple case
    #nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths = qm2.nodes_to_paths(nodes)
    rho = qm2.paths_to_gates_direct(paths, [0.1, 0.2, 0.3])
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_direct([rho])

    rho_numeric=qm2.paths_to_gates(paths)
    F_numeric=qm2.calculate_QFIM_numerical([rho_numeric], params)

    assert np.allclose(F, F_numeric, atol=1e-5), f"Direct QFIM {F} does not match numeric QFIM {F_numeric}"
    assert F.shape == (3, 3)

def test_calculate_QFIM_direct_multiple_rho():
    nodes1=[1,6,11,9,13]# Simple case
    nodes2=[4,5,10,1,13]# Simple case with Hadamard
    #nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths1 = qm2.nodes_to_paths(nodes1)
    rho1 = qm2.paths_to_gates_direct(paths1, [0.1, 0.2, 0.3])

    paths2 = qm2.nodes_to_paths(nodes2)
    rho2 = qm2.paths_to_gates_direct(paths2, [0.1, 0.2, 0.3])
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_direct([rho1,rho2])

    rho_numeric1=qm2.paths_to_gates(paths1)
    rho_numeric2=qm2.paths_to_gates(paths2)
    F_numeric=qm2.calculate_QFIM_numerical([rho_numeric1,rho_numeric2], params)

    assert np.allclose(F, F_numeric, atol=1e-5), f"Direct QFIM {F} does not match numeric QFIM {F_numeric}"
    assert F.shape == (3, 3)

def test_calculate_QFIM_direct_difficult():
    nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths = qm2.nodes_to_paths(nodes)
    rho = qm2.paths_to_gates_direct(paths, [0.1, 0.2, 0.3])
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]

    # Calculate QFIM numerically
    F = qm2.calculate_QFIM_direct([rho])
    
    assert F.shape == (3, 3)

# ============================================================================
# Test Reward Function with Numeric QFIM
# ============================================================================

def test_reward_numeric_valid():
    moves = [[2, 5, 10, 1, 13], [5, 5, 9, 1, 13], [9, 5, 9, 1, 13]]
    result = qm2.reward_numeric(moves,[0.1,0.2,0.3])
    assert isinstance(result, float)
    assert -1 <= result <= 1  # Reward should be normalized between -1 and 1

def test_reward_numeric_invalid_move():
    moves = [[1, 13, 5, 9]]  # Measurement in the middle
    result = qm2.reward_numeric(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid move should return -1

def test_reward_numeric_odd_hadamards():
    moves = [[ 1, 5, 10, 13]]  # Odd number of Hadamards
    result = qm2.reward_numeric(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid due to odd Hadamards should return -1

def test_reward_numeric_difficult():
    nodes=[[1, 2,5,2,6,7,9,9,9,10,10,13], [4,2,5,2,6,7,9,9,9,10,10,13], [7,2,5,2,6,7,9,9,9,10,10,13]]
    reward=qm2.reward_numeric(nodes,[0.1,0.2,0.3])
    assert isinstance(reward, float) or reward == -1  # Reward can be -1 for invalid moves
    assert -1 <= reward <= 1  # Reward should be normalized between -1 and

# ============================================================================
# Test Reward Function with Direct QFIM
# ============================================================================

def test_reward_direct_valid():
    moves = [[2, 5, 10, 1, 13], [5, 5, 9, 1, 13], [9, 5, 9, 1, 13]]
    result = qm2.reward_direct(moves,[0.3,0.2,0.3])
    assert isinstance(result, float)
    assert -1 <= result <= 1  # Reward should be normalized between -1 and 1

def test_reward_direct_invalid_move():
    moves = [[1, 13, 5, 9]]  # Measurement in the middle
    result = qm2.reward_direct(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid move should return -1

def test_reward_directc_odd_hadamards():
    moves = [[ 1, 5, 10, 13]]  # Odd number of Hadamards
    result = qm2.reward_direct(moves,[0.3,0.3,0.4])
    assert result == -1  # Invalid due to odd Hadamards should return -1

def test_reward_direct_difficult():
    nodes=[[1, 2,5,2,6,7,9,9,9,10,10,13], [4,2,5,2,6,7,9,9,9,10,10,13], [7,2,5,2,6,7,9,9,9,10,10,13]]
    reward=qm2.reward_direct(nodes,[0.4,0.2,0.3])
    reward_numeric=qm2.reward_numeric(nodes,[0.4,0.2,0.3])
    assert np.isclose(reward,reward_numeric, atol=1e-5)
    assert isinstance(reward, float) or reward == -1  # Reward can be -1 for invalid moves
    assert -1 <= reward <= 1  # Reward should be normalized between -1 and


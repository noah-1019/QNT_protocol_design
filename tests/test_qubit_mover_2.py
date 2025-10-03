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
# Test moves_to_gates function
# ============================================================================

def test_moves_to_gates_simple():
    moves=[1,4]
    result=qm2.moves_to_gates(moves)
    result=sp.simplify(result)


    p1, _, _ = qm2.get_all_symbols()

    expected=sp.Matrix([[1/2,1/2-p1],[1/2-p1,1/2]])



    assert result.shape == (2, 2)
    assert sp.simplify(result - expected) == sp.zeros(2, 2)

def test_moves_to_gates_with_0():
    moves=[1,0,4,0,0,0]
    result=qm2.moves_to_gates(moves)
    result=sp.simplify(result)


    p1, _, _ = qm2.get_all_symbols()

    expected=sp.Matrix([[1/2,1/2-p1],[1/2-p1,1/2]])



    assert result.shape == (2, 2)
    assert sp.simplify(result - expected) == sp.zeros(2, 2)

def test_moves_to_gates_moves_list():
    moves=[1,12,4,8,13]
    gates=qm2.nodes_to_paths(moves)

    result=qm2.moves_to_gates(gates)
    #result=sp.simplify(result) # Too complex to simplify easily

    assert result.shape == (2, 2)


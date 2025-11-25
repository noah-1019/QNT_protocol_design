"""
ciruit_builder_1.py
==========================


About:
--------------
Helper Functions for the Circuit Builder RL agent. This project is different then the previous 
qubit mover projects in that it focuses on building distribution circuits. Each step is one gate, 
this allows the agent to build a full distribution circuit from scratch. Additionally the agent
will be able to add as many qubits as it wants to the circuit, allowing for more complex circuits.

The metric that the agent will be optimizing won't be the QCRB but rather the QFIM per qubit. 
This is because the agent will be building circuits with varying numbers of qubits, so the QCRB
is not a fair metric to compare circuits with different numbers of qubits. The QFIM per qubit
is a better metric for this purpose.

To start we will be modeling the states using qutip, as it is reliable and has a lot of built in functions.

"""

# ============================================================================
# IMPORTS
# ============================================================================

### Import necessary libraries
import sys
import os

# Add parent directory to path for custom helper functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# QFIM calculations libraries
from itertools import combinations

# Estimator libraries
from scipy.optimize import root, least_squares

# Standard libraries
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Qutip libraries
from qutip import Qobj
import qutip as qt # import everything to make my life easier

# Qutip quantum circuit libraries
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import (Gate,cnot,rx,ry,rz,snot,phasegate)


# Custom helper functions
from helper_functions import qubit_mover_2 as qm2

# ============================================================================
# GLOBAL QUANTUM CONSTANTS - Initialize once, use everywhere
# ============================================================================

# Quantum Gates
# ===========================
CNOT=cnot().copy() # 2 qubit CNOT gate

# Define gates manually to ensure they have correct global phase.
X = Qobj([[0, 1], [1, 0]])
Y = Qobj([[0, -1j], [1j, 0]])
Z = Qobj([[1, 0], [0, -1]])
H = Qobj([[1, 1], [1, -1]]) / np.sqrt(2)
S = Qobj([[1, 0], [0, 1j]])
I= Qobj([[1, 0], [0, 1]])

S_DAG = S.dag().copy() # S dagger gate

# Quantum States
# ===========================
# Standard basis states

# Z basis
RHO_ZERO=qt.basis(2,0).proj().copy() # |0><0|
RHO_ONE=qt.basis(2,1).proj().copy()  # |1><1|

# X basis
RHO_PLUS=(qt.basis(2,0)+qt.basis(2,1)).unit().proj().copy()  # |+><+|
RHO_MINUS=(qt.basis(2,0)-qt.basis(2,1)).unit().proj().copy() # |-><-|

# Y basis
RHO_R=(qt.basis(2,0)+1j*qt.basis(2,1)).unit().proj().copy()  # |R><R|
RHO_L=(qt.basis(2,0)-1j*qt.basis(2,1)).unit().proj().copy() # |L><L|


# PERTURBED STATE OBJECTS
# ==========================
"""
    In order to calculate the QFIM we need to have perturbed versions of each density matrix. Using 
    Central Finite Differences we can estimate the derivatives of the density matrix with respect to
    each parameter. Thus for each density matrix we will store 6 perturbed versions, one for each
    parameter direction (pX+, pX-, pY+, pY-, pZ+, pZ-). And then one addiditional base density matrix.
    Which is the actual density matrix without any perturbations.

    Each perturbed state object is a dictionary with the following keys:
        'base': The base density matrix.
        'pX_plus': The density matrix perturbed in the +X direction.
        'pX_minus': The density matrix perturbed in the -X direction.
        'pY_plus': The density matrix perturbed in the +Y direction.
        'pY_minus': The density matrix perturbed in the -Y direction.
        'pZ_plus': The density matrix perturbed in the +Z direction.
        'pZ_minus': The density matrix perturbed in the -Z direction.

    Here we will initialize a template perturbed state object that can be copied whenever a new qubit is 
    added to the circuit.
"""

RHO_ZERO_PERTURBED={
    'base': RHO_ZERO.copy(),
    'pX_plus': RHO_ZERO.copy(),
    'pX_minus': RHO_ZERO.copy(),
    'pY_plus': RHO_ZERO.copy(),
    'pY_minus': RHO_ZERO.copy(),
    'pZ_plus': RHO_ZERO.copy(),
    'pZ_minus': RHO_ZERO.copy()
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_gates()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of all available gates for the circuit builder agent. The dict
        has the following keys: 'I', 'X', 'Y', 'Z', 'H', 'S', 'S_DAG', 'CNOT'.
    """
    return {
        'I': I.copy(),
        'X': X.copy(),
        'Y': Y.copy(),
        'Z': Z.copy(),
        'H': H.copy(),
        'S': S.copy(),
        'S_DAG': S_DAG.copy(),
        'CNOT': CNOT.copy()
    }

def get_all_basis_states()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of all available basis states for the circuit builder agent. The dict
        has the following keys: 'RHO_ZERO', 'RHO_ONE', 'RHO_PLUS', 'RHO_MINUS', 'RHO_R', 'RHO_L'.
    """
    return {
        'RHO_ZERO': RHO_ZERO.copy(),
        'RHO_ONE': RHO_ONE.copy(),
        'RHO_PLUS': RHO_PLUS.copy(),
        'RHO_MINUS': RHO_MINUS.copy(),
        'RHO_R': RHO_R.copy(),
        'RHO_L': RHO_L.copy()
    }

def get_Z_basis_states()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of Z basis states for the circuit builder agent. The dict
        has the following keys: 'RHO_ZERO', 'RHO_ONE'.
    """
    return {
        'RHO_ZERO': RHO_ZERO.copy(),
        'RHO_ONE': RHO_ONE.copy()
    }

def get_X_basis_states()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of X basis states for the circuit builder agent. The dict
        has the following keys: 'RHO_PLUS', 'RHO_MINUS'.
    """
    return {
        'RHO_PLUS': RHO_PLUS.copy(),
        'RHO_MINUS': RHO_MINUS.copy()
    }

def get_Y_basis_states()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of Y basis states for the circuit builder agent. The dict
        has the following keys: 'RHO_R', 'RHO_L'.
    """
    return {
        'RHO_R': RHO_R.copy(),
        'RHO_L': RHO_L.copy()
    }

def get_perturbed_0_state()->dict[str: Qobj]:
    """ RETURNS:
        dict: A dict of perturbed |0><0| density matrix. For more information see the RHO_ZERO_PERTURBED object.
    """

    return {
        'base': RHO_ZERO.copy(),
        'pX_plus': RHO_ZERO.copy(),
        'pX_minus': RHO_ZERO.copy(),
        'pY_plus': RHO_ZERO.copy(),
        'pY_minus': RHO_ZERO.copy(),
        'pZ_plus': RHO_ZERO.copy(),
        'pZ_minus': RHO_ZERO.copy()
    }

# ============================================================================
# APPLY GATE FUNCTIONS
# ============================================================================

# SINGLE QUBIT GATES
def apply_gate(rho: Qobj, gate: Qobj)->Qobj:
    """ Apply any given gate to a density matrix.

    ARGS:
        rho (Qobj): The input density matrix.
        gate (Qobj): The quantum gate to apply.

    RETURNS:
        Qobj: The density matrix after applying the X gate.
    """
    return gate * rho * gate.dag()

def error_gate(rho: Qobj, p_error: list[float])->Qobj:
    """ Send the qubit through a channel, applying an error to the density matrix.

    ARGS:
        rho (Qobj): The input density matrix.
        p_error list[float]: The probability of error occurring, for each type of error [p_X, p_Y, p_Z].

    RETURNS:
        Qobj: The density matrix after applying the depolarizing error channel.
    """
    p_X, p_Y, p_Z = p_error

    p_I = 1 - (p_X + p_Y + p_Z) # Probability of no error

    # Noise model as defined in the QNT paper.
    return p_I * apply_gate(rho, I) + p_X * apply_gate(rho, X) + p_Y * apply_gate(rho, Y) + p_Z * apply_gate(rho, Z)
    
# PERTURBED QUBIT FUNCTIONS
def generate_entanglement_rho_dict(rho_dict: dict[str: Qobj])-> dict[str: Qobj]:
    """ Generate entanglement between two qubits, one given by rho_dict and the other
        in the |0> state.

    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.

    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after generating entanglement.
    """
    
    new_rho_dict = {
            'base': CNOT * qt.tensor(rho_dict['base'], RHO_ZERO) * CNOT.dag(),
            'pX_plus': CNOT * qt.tensor(rho_dict['pX_plus'], RHO_ZERO) * CNOT.dag(),
            'pX_minus': CNOT * qt.tensor(rho_dict['pX_minus'], RHO_ZERO) * CNOT.dag(),
            'pY_plus': CNOT * qt.tensor(rho_dict['pY_plus'], RHO_ZERO) * CNOT.dag(),
            'pY_minus': CNOT * qt.tensor(rho_dict['pY_minus'], RHO_ZERO) * CNOT.dag(),
            'pZ_plus': CNOT * qt.tensor(rho_dict['pZ_plus'], RHO_ZERO) * CNOT.dag(),
            'pZ_minus': CNOT * qt.tensor(rho_dict['pZ_minus'], RHO_ZERO) * CNOT.dag()
    }

    return new_rho_dict
    
def apply_gate_rho_dict(rho_dict: dict[str: Qobj], gate: Qobj)-> dict[str: Qobj]:
    """ Apply a quantum gate to all density matrices in a dictionary.

    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.
        gate (Qobj): The quantum gate to apply.

    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after applying the gate.
    """
    return {key: apply_gate(rho, gate) for key, rho in rho_dict.items()}

def error_gate_perturbed(rho_dict: dict[str: Qobj], p_error: list[float], h: float = 1e-6)-> dict[str: Qobj]:
    """ Send the qubit through a channel, applying an error to the density matrix.
        This function perturbs the error probabilities by a small amount delta. This
        is useful for estimating the QFIM. 

    ARGS:
        rho_list dict[str: Qobj]: The input density matrices.
        p_error list[float]: The probability of error occurring, for each type of error [p_X, p_Y, p_Z].
        h (float): The small perturbation to apply to each error probability.

    RETURNS:
        dict[str:Qobj]: A dictionary containing the base density matrix and the perturbed density matrices.
    """

    p_X, p_Y, p_Z = p_error

    # Perturbated paramater tuples:

    # Perturb the error probabilities
    p_X_perturbed = (min(max(p_X + h, 0), 1),min(max(p_X - h, 0), 1)) # Ensure probabilities remain valid
    p_Y_perturbed = (min(max(p_Y + h, 0), 1),min(max(p_Y - h, 0), 1))
    p_Z_perturbed = (min(max(p_Z + h, 0), 1),min(max(p_Z - h, 0), 1))

    
    # Noise model as defined in the QNT paper.
    rho_base=error_gate(rho_dict['base'], p_error) # Actually density matrix

    # Perturbations in P_Z direction
    rho_pX_plus = error_gate(rho_dict['pX_plus'], [p_X_perturbed[0], p_Y, p_Z])
    rho_pX_minus = error_gate(rho_dict['pX_minus'], [p_X_perturbed[1], p_Y, p_Z])

    # Perturbations in P_Y direction
    rho_pY_plus = error_gate(rho_dict['pY_plus'], [p_X, p_Y_perturbed[0], p_Z])
    rho_pY_minus = error_gate(rho_dict['pY_minus'], [p_X, p_Y_perturbed[1], p_Z])

    # Perturbations in P_Z direction
    rho_pZ_plus = error_gate(rho_dict['pZ_plus'], [p_X, p_Y, p_Z_perturbed[0]])
    rho_pZ_minus = error_gate(rho_dict['pZ_minus'], [p_X, p_Y, p_Z_perturbed[1]])

    return {
        'base': rho_base,
        'pX_plus': rho_pX_plus,
        'pX_minus': rho_pX_minus,
        'pY_plus': rho_pY_plus,
        'pY_minus': rho_pY_minus,
        'pZ_plus': rho_pZ_plus,
        'pZ_minus': rho_pZ_minus
    }

# MULTI QUBIT GATES
def error_gate_tensor(rho: Qobj, 
                      p_error: list[float],
                      qubit_id: int,
                      entanglement_vector: list[int])->Qobj:

    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply gate directly
        return error_gate(rho, p_error)
    p_X, p_Y, p_Z = p_error

    p_I = 1 - (p_X + p_Y + p_Z) # Probability of no error

    return (p_I * multi_qubit_apply_gate(rho, I,qubit_id,entanglement_vector) + 
           p_X * multi_qubit_apply_gate(rho, X,qubit_id,entanglement_vector) + 
           p_Y * multi_qubit_apply_gate(rho, Y,qubit_id,entanglement_vector) + 
           p_Z * multi_qubit_apply_gate(rho, Z,qubit_id,entanglement_vector) 
           )

def multi_qubit_apply_gate(rho: Qobj, 
                           gate: Qobj,
                           qubit_id: int,
                           entanglement_vector: list[int])->Qobj:

    """ Apply a quantum gate to a specific qubit in a multi-qubit density matrix.
    ARGS:
        rho (Qobj): The input density matrix.
        gate (Qobj): The quantum gate to apply.
        qubit_index (int): The index of the qubit to apply the gate to.
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.

    RETURNS:
        Qobj: The density matrix after applying the gate.
    """

    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply gate directly
        return(gate * rho * gate.dag())


    filtered_entanglement_vector = entanglement_vector.copy()
    filtered_entanglement_vector.append(qubit_id) # Add the target qubit
    filtered_entanglement_vector.sort() 
    # Sort the vector to maintain order this is the order the tensor product of the state
    # was built in.

    target=[filtered_entanglement_vector.index(qubit_id)] # find the index of the qubit in the state
    num_qubits = len(filtered_entanglement_vector)-1 # The total number of qubits in the system

    dims=[2 for _ in range(num_qubits)] # Dimension list for all qubits
    expanded_gate = qt.expand_operator(oper=gate, dims=dims, targets=target) # Expand the gate to the full system

    return expanded_gate * rho * expanded_gate.dag() # Apply the gate to the density matrix.

def generate_entanglement_multi_qubit(rho: Qobj,
                                      qubit_id: int,
                                      entanglement_vector: list[int])->Qobj:
    """ Generate entanglement between two states, one given by rho and the other
        in the |0> state.
    ARGS:
        rho (Qobj): The input density matrix.
        qubit_index (int): The index of the qubit to apply the gate to.
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.

    RETURNS:
        Qobj: The density matrix after generating entanglement.
    """

    CNOT_gate= get_all_gates()['CNOT']
    ket0=get_Z_basis_states()['RHO_ZERO']

    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply CNOT directly
        return(CNOT_gate * qt.tensor(rho, ket0) * CNOT_gate.dag())


    filtered_entanglement_vector = entanglement_vector.copy()
    filtered_entanglement_vector.append(qubit_id) # Add the target qubit
    filtered_entanglement_vector.sort() 

    num_qubits = len(filtered_entanglement_vector) # The total number of qubits in the system
    target=[filtered_entanglement_vector.index(qubit_id)] # find the index of the qubit in the state
    target.append(num_qubits-1)  # The new qubit is always the last qubit, so target the last qu

    # represent the state in an expanded hilbert space

    expanded_rho = qt.tensor(rho, ket0) # Add the |0><0| state

    dims=[2 for _ in range(num_qubits)] # Dimension list for all qubits
    expanded_CNOT = qt.expand_operator(oper=CNOT_gate, dims=dims, targets = target)

    return expanded_CNOT * expanded_rho * expanded_CNOT.dag() # Apply the CNOT gate to the density matrix.


# MULTI QUBIT PERTURBED FUNCTIONS
def multi_qubit_apply_gate_dict(rho_dict: dict[str: Qobj],
                           gate: Qobj, 
                           qubit_id: int, 
                           entanglement_vector: list[int],
                           ):
    """ Apply a quantum gate to a specific qubit in a multi-qubit density matrix.
    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.
        gate (Qobj): The quantum gate to apply.
        qubit_index (int): The index of the qubit to apply the gate to.
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.
    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after applying the gate.

    """
    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply gate directly
        return {key: gate * rho * gate.dag() for key, rho in rho_dict.items()}

    filtered_entanglement_vector = entanglement_vector.copy()
    filtered_entanglement_vector.append(qubit_id) # Add the target qubit
    filtered_entanglement_vector.sort() 
    # Sort the vector to maintain order this is the order the tensor product of the state
    # was built in.

    target=[filtered_entanglement_vector.index(qubit_id)] # find the index of the qubit in the state

    num_qubits = len(filtered_entanglement_vector)-1 # The total number of qubits in the system
    
    dims=[2 for _ in range(num_qubits)] # Dimension list for all qubits
    n_gate = qt.expand_operator(oper=gate, dims=dims, targets=target) # Expand the gate to the full system

    return {key: n_gate * rho * n_gate.dag() for key, rho in rho_dict.items()} # Apply the gate to all density matrices in the dictionary.

def multi_qubit_error_gate(rho_dict: dict[str: Qobj],
                           parameters: list[float], 
                           qubit_id: int,
                           entanglement_vector: list[int],
                           h: float = 1e-6
                           )-> dict[str: Qobj]:
    
    """ Apply a quantum error channel to a specific qubit in a multi-qubit density matrix.
    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.
        parameters (list[float]): The error parameters [p_X, p_Y, p_Z].
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.
        qubit_index (int): The index of the qubit to apply the error channel to.

    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after applying the error channel.

    """
    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply gate directly
        return error_gate_perturbed(rho_dict, parameters, h)

    filtered_entanglement_vector = [i for i in entanglement_vector if i != -1] # Remove -1 entry
    filtered_entanglement_vector.append(qubit_id) # Add the target qubit
    filtered_entanglement_vector.sort() 

    p_X, p_Y, p_Z = parameters

    # Perturbated paramater tuples:

    # Perturb the error probabilities
    p_X_perturbed = (min(max(p_X + h, 0), 1),min(max(p_X - h, 0), 1)) # Ensure probabilities remain valid
    p_Y_perturbed = (min(max(p_Y + h, 0), 1),min(max(p_Y - h, 0), 1))
    p_Z_perturbed = (min(max(p_Z + h, 0), 1),min(max(p_Z - h, 0), 1))


    rho_base=error_gate_tensor(rho_dict['base'], parameters,qubit_id,entanglement_vector) # Actually density matrix

    # Perturbations in P_Z direction
    rho_pX_plus = error_gate_tensor(rho_dict['pX_plus'], [p_X_perturbed[0], p_Y, p_Z],qubit_id,entanglement_vector)
    rho_pX_minus = error_gate_tensor(rho_dict['pX_minus'], [p_X_perturbed[1], p_Y, p_Z],qubit_id,entanglement_vector)

    # Perturbations in P_Y directions
    rho_pY_plus = error_gate_tensor(rho_dict['pY_plus'], [p_X, p_Y_perturbed[0], p_Z],qubit_id,entanglement_vector)
    rho_pY_minus = error_gate_tensor(rho_dict['pY_minus'], [p_X, p_Y_perturbed[1], p_Z],qubit_id,entanglement_vector)

    # Perturbations in P_Z direction
    rho_pZ_plus = error_gate_tensor(rho_dict['pZ_plus'], [p_X, p_Y, p_Z_perturbed[0]],qubit_id,entanglement_vector)
    rho_pZ_minus = error_gate_tensor(rho_dict['pZ_minus'], [p_X, p_Y, p_Z_perturbed[1]],qubit_id,entanglement_vector)

    return {
        'base': rho_base,
        'pX_plus': rho_pX_plus,
        'pX_minus': rho_pX_minus,
        'pY_plus': rho_pY_plus,
        'pY_minus': rho_pY_minus,
        'pZ_plus': rho_pZ_plus,
        'pZ_minus': rho_pZ_minus
    }

def multi_qubit_error_gate_non_perturbed(rho_dict: dict[str: Qobj],
                                        parameters: list[float], 
                                        qubit_id: int,
                                        entanglement_vector: list[int],
                                        )-> dict[str: Qobj]:
    """ Apply a quantum error channel to a specific qubit in a multi-qubit density matrix.
    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.
        parameters (list[float]): The error parameters [p_X, p_Y, p_Z].
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.
        qubit_index (int): The index of the qubit to apply the error channel to.

    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after applying the error channel.

    """
    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply gate directly
        return {key: error_gate(rho, parameters) for key, rho in rho_dict.items()}

    return {
        key: error_gate_tensor(rho, parameters,qubit_id,entanglement_vector) for key, rho in rho_dict.items()
    }

def generate_entanglement_multi_qubit_dict(rho_dict: dict[str: Qobj],
                                           qubit_id: int,
                                           entanglement_vector: list[int]
                                           )-> dict[str: Qobj]:
    """ Generate entanglement between two states, one given by rho_dict and the other
        in the |0> state.
    ARGS:
        rho_dict (dict[str: Qobj]): A dictionary of density matrices.
        qubit_index (int): The index of the qubit to apply the gate to.
        entanglement_vector (list[int]): A list of qubit indices that are entangled with the target qubit.
    RETURNS:
        dict[str: Qobj]: A dictionary of density matrices after generating entanglement.
    """

    if len(entanglement_vector)==1: # If there are no entangled qubits, just apply CNOT directly
        return {
            key: CNOT * qt.tensor(rho, get_Z_basis_states()['RHO_ZERO']) * CNOT.dag() for key, rho in rho_dict.items()
        }

    gates=get_all_gates()
    CNOT_gate=gates['CNOT']
    ket0=get_Z_basis_states()['RHO_ZERO']

    filtered_entanglement_vector = [i for i in entanglement_vector if i != -1] # Remove -1 entries
    filtered_entanglement_vector.append(qubit_id) # Add the target qubit
    filtered_entanglement_vector.sort()
    target=[filtered_entanglement_vector.index(qubit_id)] # find the index of the qubit in the state
    


    num_qubits = len(filtered_entanglement_vector) # The total number of qubits in the system
    target.append(num_qubits-1)

    # represent the state in an expanded hilbert space
    expanded_rho_dict = {
        key: qt.tensor(rho, ket0) for key, rho in rho_dict.items()
    } # Add the |0><0| state

    dims=[2 for _ in range(num_qubits)] # Dimension list for all qubits
    expanded_CNOT = qt.expand_operator(oper=CNOT_gate, dims=dims, targets = target)

    expanded_CNOT_dag = expanded_CNOT.dag() # Precompute the dagger of the CNOT gate

    return {
        key: expanded_CNOT * rho_expanded * expanded_CNOT_dag for key, rho_expanded in expanded_rho_dict.items()
    } # Apply the CNOT gate to all density matrices in the dictionary.



# ============================================================================
# ACTION ENCODING FUNCTION ~~~~~~ DEPRECATED ~~~~~~
# ============================================================================

def actions_to_gates(prior_action_dict: dict[int: [dict[str: Qobj],list[int],int]],
                     current_action_vector: list[int],
                     parameters: list[list[int]],
                     DEBUG: bool = False
                     )->dict[int: dict[str: Qobj],list[int],int]:
    """ Convert a list of action indices to a list of quantum gates.

    Action Encoding
    --------------------------
    The action for each move is one hot encoded as a vector of the following type:    
        [action_type, gate_id, qubit_id, end_loc, source_loc]

    There are 5 possible action types: 
        0: Apply Gate
        1: Move Qubit
        2: Add Qubit
        3: Generate Entanglement
        4: Measure All Qubits

    Each action type will use an action mask to only utilize the relevant parts of the action vector.
    0: Apply Gate
        gate_id: The gate to apply (0: I, 1: X, 2: Y, 3: Z, 4: H, 5: S 6: S_DAG)
        qubit_id: The qubit to apply the gate to
    1: Move Qubit
        qubit_id: The qubit to move
        end_loc: The location to move the qubit to (If the same as the current location the qubit is mvoed 
                 to the center and then back to the original location) If the qubit picks node 0 and is 
                 already there it will not move.
    2: Add Qubit
        source_loc: The location to add the qubit from.
    3: Generate Entanglement
        qubit_id: The qubit to entangle with a new qubit. This will be generated by a CNOT wherever the 
                  qubit currently is.
    4: Measure All Qubits
        No additional parameters needed. All qubits are measured. If not at an end node they are moved to 
        end node 1. 

        
    Thus the vector type is:
        0: action_type: either 0, 1, 2, 3, or 4
        1: gate_id: integer from 0 to 6
        2: qubit_id: integer from 0 to (num_qubits - 1)
        3: end_loc: integer from 0 to 3
        4: source_loc: integer from 1 to 3 (0 is not allowed as a source location)
    

    Notes
    -----
    The first action will be to add one qubit and the location will be given by the source location parameter.
    ARGS:
        prior_action_dict (dict[int: dict[str: Qobj],list[int],int]): 
        A dictionary containing the quantum circuit information for each qubit.
            The keys are the qubit IDs and the values are a list containing:
                0: dict[str: Qobj]: The density matrices for the qubit.
                1: list[int]: The entanglement partners of the qubit.
                2: int: The current location of the qubit.
        current_action_vector (list[int]): A list representing the current action to take.
        DEBUG (bool): Whether to print debug information.

    RETURNS:
        dict[int: [dict[str: Qobj],list[int],int]]: 
        A dictionary containing the updated quantum circuit information for each qubit. The format is the same as prior_action_dict. 
        
    """
    
    # Get operations
    gate_dict = get_all_gates()
    gates = [gate_dict['I'], 
             gate_dict['X'], 
             gate_dict['Y'], 
             gate_dict['Z'], 
             gate_dict['H'], 
             gate_dict['S'],
             gate_dict['S_DAG']]
    # The gate id is the index of this list.
    
    # Get network info
    channel_1_params = parameters[0] # Parameters for channel 1
    channel_2_params = parameters[1] # Parameters for channel 2
    channel_3_params = parameters[2] # Parameters for channel 3

    # Initialize return dict
    return_dict = prior_action_dict.copy() # Copy prior actions

    # Unpack action vector so I don't go crazy:
    action_type = current_action_vector[0]
    gate_id = current_action_vector[1]
    qubit_id = current_action_vector[2]
    end_loc = current_action_vector[3]
    source_loc = current_action_vector[4]


    # Initialize the quantum circuit
    #--------------------------------------------
    num_qubits = len(prior_action_dict)
    if DEBUG: 
        print(f"Number of qubits: {num_qubits}")
        print("Action Vector: ", current_action_vector)

    # Special case for the first action when there are no qubits yet:
    if num_qubits == 0:
        if DEBUG:
            print("No qubits, so first action must be to add a qubit.")

        rho_dict={ # Initialize the density matrix dict for the new qubit
            'base': RHO_ZERO.copy(),
            'pX_plus': RHO_ZERO.copy(),
            'pX_minus': RHO_ZERO.copy(),
            'pY_plus': RHO_ZERO.copy(),
            'pY_minus': RHO_ZERO.copy(),
            'pZ_plus': RHO_ZERO.copy(),
            'pZ_minus': RHO_ZERO.copy()
        }

        return_dict[0] = [rho_dict, [0],source_loc]  # Add the new qubit to the return dict
        # Start with qubit in |0> state not entangled with anyone. At the 
        # Specified source location.
        return return_dict
    
    # Otherwise proceed as normal:

    #--------------------------------------------------------------
    # ACTION TYPE 0: APPLY GATE
    #--------------------------------------------------------------
    if action_type == 0: # Apply Gate
        if DEBUG:
            print(f"Applying gate {gate_id} to qubit {qubit_id}.")
            print(f"Location of qubit {qubit_id}: {prior_action_dict[qubit_id][2]}")

        # Get the gate to apply
        gate_to_apply = gates[gate_id]

        # Apply the gate to the specified qubit
        prior_rho_dict = prior_action_dict[qubit_id][0]
        new_rho_dict = multi_qubit_apply_gate_dict(prior_rho_dict, 
                                              gate_to_apply, 
                                              qubit_id, 
                                              prior_action_dict[qubit_id][1])

        # Update the return dict
        for partner_id in prior_action_dict[qubit_id][1]:
                    return_dict[partner_id] = [new_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # The location remains the same

        return return_dict


    #--------------------------------------------------------------
    # ACTION TYPE 1: MOVE QUBIT
    #--------------------------------------------------------------
    elif action_type == 1: # Move Qubit
        # This action requires some logic. There are 5 cases to consider:
        # 1. The qubit is already at the desired end location not in center: 
        #   In this case the qubit is moved to the center and then back to the original location.

        # 2. The qubit is at the center and the desired end location is not the center:
        #   In this case the qubit is moved to the desired end location.

        # 3. The qubit is at an edge location and the desired end location is the center:
        #   In this case the qubit is moved to the center.

        # 4. The qubit is in the center and the desired end location is also the center:
        #   In this case the qubit remains in the center.
    
        # 5. The qubit is at an edge location and the desired end location is a different edge location:
        #   In this case the qubit is moved to the center and then to the desired end location.

        # Determine current location of the qubit
        current_loc = prior_action_dict[qubit_id][2]
        if DEBUG:
            print(f"Moving qubit {qubit_id} from location {current_loc} to {end_loc}.")

        # Get the prior density matrix dict
        prior_rho_dict = prior_action_dict[qubit_id][0]
        entanglement_vector = prior_action_dict[qubit_id][1]

        # Case 1: the qubit is already at the desired end location not in center
        # -----------------------------------------------------------------------
        if current_loc == end_loc and current_loc != 0:
            if DEBUG:
                print(f"Qubit {qubit_id} is already at location {end_loc}, moving to center and back.")

            # Move to center
            # Determine channel parameters for the move to center
            if end_loc == 1:
                channel_params = channel_1_params
            elif end_loc == 2:
                channel_params = channel_2_params
            elif end_loc == 3:
                channel_params = channel_3_params
            else:
                raise ValueError("Invalid end location.")
            
            if DEBUG:
                print("Channel params: ", channel_params)

            intermediate_rho_dict = multi_qubit_error_gate(prior_rho_dict, 
                                                           channel_params,
                                                           qubit_id,
                                                           entanglement_vector
                                                           ) 

            # Move back to original location
            final_rho_dict = multi_qubit_error_gate(intermediate_rho_dict, 
                                                           channel_params,
                                                           qubit_id,
                                                           entanglement_vector
                                                           )

            # Update the return dict 
            # We must update the entanglement partners as well
            for partner_id in prior_action_dict[qubit_id][1]:
                    return_dict[partner_id] = [final_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # The location remains the same

            return return_dict
        
        # Case 2: The Qubit is moved from center to an end location
        # -----------------------------------------------------------------------
        elif current_loc == 0 and end_loc != 0:
            if DEBUG:
                print(f"Qubit {qubit_id} is at center, moving to location {end_loc}.")

            # Move to desired end location
            # Determine channel parameters for the move to end location
            if end_loc == 1:
                channel_params = channel_1_params
            elif end_loc == 2:
                channel_params = channel_2_params
            elif end_loc == 3:
                channel_params = channel_3_params
            else:
                raise ValueError("Invalid end location.")
            
            final_rho_dict = multi_qubit_error_gate(prior_rho_dict, 
                                                           channel_params,
                                                           qubit_id,
                                                           entanglement_vector,
                                                           ) 

            # Update the location
            return_dict[qubit_id][2] = end_loc # The new state after move
            # Update the return dict 
            # We must update the entanglement partners as well
            for partner_id in prior_action_dict[qubit_id][1]:
                    return_dict[partner_id] = [final_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # The location remains the same
            
            return return_dict
        
        # Case 3: The Qubit is moved from an end location to the center
        # -----------------------------------------------------------------------
        elif current_loc != 0 and end_loc == 0:
            if DEBUG:
                print(f"Qubit {qubit_id} is at location {current_loc}, moving to center.")

            # Move to center
            # Determine channel parameters for the move to center
            if current_loc == 1:
                channel_params = channel_1_params
            elif current_loc == 2:
                channel_params = channel_2_params
            elif current_loc == 3:
                channel_params = channel_3_params
            else:
                raise ValueError("Invalid current location.")
            
            final_rho_dict = multi_qubit_error_gate(prior_rho_dict, 
                                                           channel_params,
                                                           qubit_id,
                                                           entanglement_vector
                                                           ) 

            # Update the location
            return_dict[qubit_id][2] = 0 # The new state after move

            # Update the return dict 
            # We must update the entanglement partners as well
            for partner_id in prior_action_dict[qubit_id][1]:
                    return_dict[partner_id] = [final_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # The location remains the same

            return return_dict

        # Case 4: The Qubit is at center and end location is also center
        # -----------------------------------------------------------------------
        elif current_loc == 0 and end_loc == 0:
            if DEBUG:
                print(f"Qubit {qubit_id} is already at center, no move needed.")

            # No move needed, just copy prior state
            return_dict[qubit_id] = [prior_rho_dict, # The state remains the same
                                     prior_action_dict[qubit_id][1], # The entanglement partners remain the same
                                     current_loc] # The location remains the same
            
            return return_dict
        
        # Case 5: The qubit is at an edge location and 
        # the desired end location is a different edge location
        # --------------------------------------------------------
        elif current_loc != 0 and end_loc != 0 and current_loc != end_loc:
            if DEBUG:
                print(f"Qubit {qubit_id} is at location {current_loc}, moving to center and then to location {end_loc}.")

            # Move to center
            # Determine channel parameters for the move to center
            if current_loc == 1:
                channel_params_to_center = channel_1_params
            elif current_loc == 2:
                channel_params_to_center = channel_2_params
            elif current_loc == 3:
                channel_params_to_center = channel_3_params
            else:
                raise ValueError("Invalid current location.")
            
            intermediate_rho_dict = multi_qubit_error_gate(prior_rho_dict, 
                                                           channel_params_to_center,
                                                           qubit_id,
                                                           entanglement_vector) 

            # Move to desired end location
            # Determine channel parameters for the move to end location
            if end_loc == 1:
                channel_params_to_end = channel_1_params
            elif end_loc == 2:
                channel_params_to_end = channel_2_params
            elif end_loc == 3:
                channel_params_to_end = channel_3_params
            else:
                raise ValueError("Invalid end location.")
            
            final_rho_dict = multi_qubit_error_gate(intermediate_rho_dict, 
                                                           channel_params_to_end,
                                                            qubit_id,
                                                           entanglement_vector
                                                           )


            # Update the location
            return_dict[qubit_id][2] = end_loc # The new state after move
            # Update the return dict 
            # We must update the entanglement partners as well
            for partner_id in prior_action_dict[qubit_id][1]:
                    return_dict[partner_id] = [final_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # The location remains the same

            return return_dict
        
        # Else: Throw an error
        else:
            raise ValueError("Invalid move action parameters.")

    #--------------------------------------------------------------
    # ACTION TYPE 2: ADD QUBIT
    #--------------------------------------------------------------
    elif action_type == 2: # Add Qubit
        if DEBUG:
            print(f"Adding new qubit at location {source_loc}.")

        # Initialize the density matrix dict for the new qubit
        rho_dict={ 
            'base': RHO_ZERO.copy(),
            'pX_plus': RHO_ZERO.copy(),
            'pX_minus': RHO_ZERO.copy(),
            'pY_plus': RHO_ZERO.copy(),
            'pY_minus': RHO_ZERO.copy(),
            'pZ_plus': RHO_ZERO.copy(),
            'pZ_minus': RHO_ZERO.copy()
        }

        new_qubit_id = max(prior_action_dict.keys()) + 1 # New qubit ID

        return_dict[new_qubit_id] = [rho_dict, # The new qubit state
                                     [new_qubit_id],     # Not entangled with anyone
                                     source_loc]  # At the specified source location

    #--------------------------------------------------------------
    # ACTION TYPE 3: GENERATE ENTANGLEMENT
    #--------------------------------------------------------------
    elif action_type == 3: # Generate Entanglement
        if DEBUG:
            print(f"Generating entanglement for qubit {qubit_id}.")

        # Get the prior density matrix dict
        prior_rho_dict = prior_action_dict[qubit_id][0]

        # Apply CNOT gate to generate entanglement
        # For simplicity, we will assume the new qubit is initialized in |0> state
        new_rho_dict = generate_entanglement_multi_qubit_dict(prior_rho_dict,
                                                              qubit_id,
                                                              prior_action_dict[qubit_id][1])

        # Update entanglement partners
        prior_partners = prior_action_dict[qubit_id][1]
        new_qubit_id = max(prior_action_dict.keys()) + 1 # New qubit ID
        new_partners = prior_partners+[new_qubit_id] # Add new qubit to partners

        if DEBUG:
            print(f"New qubit ID: {new_qubit_id}")
            print(f"Updated entanglement partners for qubit {qubit_id}: {new_partners}")

        # Update the return dict for the original qubits
        for temp_qubit_id in prior_partners:
            return_dict[temp_qubit_id] = [new_rho_dict, # The new state after entanglement
                                        new_partners, # Updated entanglement partners
                                        prior_action_dict[temp_qubit_id][2]] # The location remains the same


        # update the return dict for the new qubit
        return_dict[new_qubit_id] = [new_rho_dict, # The new qubit state
                                     new_partners,  # Entangled with the original qubit
                                     prior_action_dict[qubit_id][2]]  # At the same location as the original qubit

        
        return return_dict


    #--------------------------------------------------------------
    # ACTION TYPE 4: MEASURE ALL QUBITS
    #--------------------------------------------------------------
    elif action_type == 4: # Measure All Qubits
        # This requires moving all qubits to end node 1 if they are in the center.
        if DEBUG:
            print("Measuring all qubits, moving to end node 1 if necessary.")

        for q_id in prior_action_dict:
            current_loc = prior_action_dict[q_id][2]
            prior_rho_dict = prior_action_dict[q_id][0]

            if current_loc == 0:
                if DEBUG:
                    print(f"Moving qubit {q_id} from center to end node 1.")

                
                final_rho_dict = multi_qubit_error_gate(prior_rho_dict, 
                                                        channel_1_params,
                                                        q_id,
                                                        prior_action_dict[q_id][1])

                # Update the location
                return_dict[q_id][2]=1
                for partner_id in prior_action_dict[q_id][1]:
                    return_dict[partner_id] = [final_rho_dict, # The new state after move
                                                prior_action_dict[partner_id][1], # The entanglement partners remain the same
                                                prior_action_dict[partner_id][2]] # Update to end node 1 location
            else:
                if DEBUG:
                    print(f"Qubit {q_id} is already at end node 1, no move needed.")
                
                # No move needed, just copy prior state
                return_dict[q_id] = [prior_rho_dict, # The state remains the same
                                     prior_action_dict[q_id][1], # The entanglement partners remain the same
                                     current_loc] # The location remains the same


    else:        
        raise ValueError("Invalid action type.")

    return return_dict
    
def get_test_action_dict():
    """
    This function is used to generate a test action dict for debugging purposes. 
    Generate a test action dict with 3 qubits:
        Qubit 0: |0> state at node 1
        Qubit 1: |+> state at node 2
        Qubit 2 and 3: Entangled pair at node 3
    RETURNS:
        dict[int: [dict[str: Qobj],list[int],int]]: 
        A dictionary containing the quantum circuit information for each qubit.
    """
     # Qubit 0: |0> state at node 1
    qubit0=get_perturbed_0_state()
    qubit1=get_perturbed_0_state()
    qubit1=multi_qubit_apply_gate_dict(qubit1,get_all_gates()['H'],0,[0])
    qubit2=get_perturbed_0_state()
    qubit2=multi_qubit_apply_gate_dict(qubit2,get_all_gates()['S'],0,[0])
    ent_qubit=generate_entanglement_multi_qubit_dict(qubit2,2,[2])

    return {0: [qubit0, [0], 1],
            1: [qubit1, [1], 2],
            2: [ent_qubit, [2,3], 3],
            3: [ent_qubit, [2,3], 0]}

def get_test_action_dict_entangled():
    """
    This function is used to generate a test action dict for debugging purposes. 
    Generate a test action dict with 2 qubits entangled:
        Qubit 0 and 1: Entangled pair at node 2 and 0 respectively. Both
        are in the |0> state initially.
    RETURNS:
        dict[int: [dict[str: Qobj],list[int],int]]: 
        A dictionary containing the quantum circuit information for each qubit.
    """
     # Qubit 0 and 1: Entangled pair at node 2 and 0 respectively.
    qubit0=get_perturbed_0_state()
    ent_qubit=generate_entanglement_multi_qubit_dict(qubit0,0,[0])

    return {0: [ent_qubit, [0,1], 2],
            1: [ent_qubit, [0,1], 0]}


# ============================================================================
# Calculate QFIM Functions
# ============================================================================
def calculate_QFIM_direct_singular(rho_dict_list: list[dict[str: Qobj]],
                                   h=1e-5,
                                   debug: bool = False):
    """
    Calculate QFIM for a single density matrix using direct numerical methods.
    
    This function computes the QFIM using pre-computed perturbed density matrices
    and numerical differentiation. It's designed for maximum efficiency in RL
    environments where the same perturbations are used repeatedly.

    The formula used is:
    
    F_ab = sum over i,j=0 to d-1 (with λ_i + λ_j ≠ 0) of:
        [ 2 * Re( ⟨λ_i|∂_aρ|λ_j⟩ * ⟨λ_j|∂_bρ|λ_i⟩ ) ] / (λ_i + λ_j)


    """

    # # Calculate the eigenvalues and eigenvectors of each density matrix
    # # =========================================================================

    # Calcualate base eigenvalues and eigenvectors
    rho_base= rho_dict_list[0]['base'] # The. base density matrix should be the same for all channels

    if debug:
        print("Base Density Matrix: \n", rho_base.full())

    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(rho_base.full())

    if debug:
        print("Eigenvalues: \n", eigenvals)
        print("Eigenvectors: \n", eigenvecs)
    
    
    # Calculate the derivatives of rho with respect to each parameter
    # =========================================================================

    derivatives= [
        [[],[],[]],
        [[],[],[]],
        [[],[],[]]
                  ]
    # This list will hold the derivatives of the density matrix with respect to each parameter
    # The first index represents the channel (0,1,2)
    # The second index represents the parameter (0:X,1:Y,2:Z)
    # The third index represents the eigenvalue index

    for channel_id in range(3): # For each channel
        for param_id in range(3): # For each parameter
            # Get the perturbed density matrices
            rho_plus = rho_dict_list[channel_id][f'p{"XYZ"[param_id]}_plus']
            rho_minus = rho_dict_list[channel_id][f'p{"XYZ"[param_id]}_minus']

            # Calculate the derivative using central difference
            # this approach utilizes numpy magic
            derivative_matrix = (rho_plus.full() - rho_minus.full()) / (2 * h)

            if debug:
                print(f"Channel {channel_id+1} Parameter {'XYZ'[param_id]} Derivative Matrix: \n", derivative_matrix)

            derivatives[channel_id][param_id] = derivative_matrix

    
    
    # Compute the matrix
    n_params = 9 # We are looking at 9 parameters total (3 per channel)

    QFIM = np.zeros((n_params, n_params))

    for a in range(n_params):
        for b in range(n_params):
            sum_ab = 0.0
            for i in range(len(eigenvals)): # Sum over all eigenvalues
                for j in range(len(eigenvals)): # Sum over all eigenvalues
                    lambda_i = eigenvals[i]
                    lambda_j = eigenvals[j]
                    if lambda_i + lambda_j != 0: # Avoid divergent terms
                        # Determine channel and parameter indices
                        
                        # These are the indices for the derivative matrices
                        channel_a = a // 3
                        param_a = a % 3
                        channel_b = b // 3
                        param_b = b % 3

                        # Get the derivatives
                        d_rho_a = derivatives[channel_a][param_a]
                        d_rho_b = derivatives[channel_b][param_b]

                        # Calculate the matrix elements
                        bra_i = eigenvecs[:, i].conj().T
                        # The conjugate is not necessary since
                        # the matrix is Hermitian but I include it for clarity
                        ket_j = eigenvecs[:, j]
                        bra_j = eigenvecs[:, j].conj().T
                        ket_i = eigenvecs[:, i]

                        element_a = bra_i @ d_rho_a @ ket_j
                        element_b = bra_j @ d_rho_b @ ket_i

                        sum_ab += (2 * np.real(element_a * element_b)) / (lambda_i + lambda_j)
            QFIM[a, b] += sum_ab

    
    return QFIM


#===========================================================================
# Calculate Entanglement Entropy Function
#===========================================================================

def calculate_entanglement_entropy(rho: Qobj,
                                   qubit_id: int,
                                   entanglement_vector: list[int]
                                   ) -> float:
    
    """
    Calculate the entanglement entropy of a qubit with its entanglement partners.
    ARGS:
        rho (Qobj): The density matrix of the entire system.
        qubit_id (int): The ID of the qubit to calculate the entanglement entropy for.
        entanglement_vector (list[int]): The list of qubit IDs that are entangled with the target qubit.

    RETURNS:
        float: The entanglement entropy of the target qubit.
    """

    # Get the index of the qubit in the density matrix
    index = entanglement_vector.index(qubit_id)
    
    # Trace out all other qubits to get the reduced density matrix
    p_rho = rho.ptrace(index)

    # Calculate the entropy between the qubit and its entanglement partners
    entropy = qt.entropy_vn(p_rho, base=2)

    return entropy 
    

# ===========================================================================
# Visualization functions
# ===========================================================================

def plot_complex_array(arr):
    """
    Plots a complex NumPy square array in 3D.
    - X and Y: array indices
    - Z: real part of the complex number
    - Color: imaginary part of the complex number
    """
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be a square 2D NumPy array.")

    n = arr.shape[0]
    x, y = np.meshgrid(range(n), range(n))
    z = arr.real
    colors = arr.imag

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize colors for colormap
    norm = plt.Normalize(colors.min(), colors.max())
    cmap = plt.cm.viridis

    ax.bar3d(x.flatten(), y.flatten(), np.zeros_like(z.flatten()),
             dx=0.8, dy=0.8, dz=z.flatten(),
             color=cmap(norm(colors.flatten())), shade=True)

    ax.set_xlabel('X index')
    ax.set_ylabel('Y index')
    ax.set_zlabel('Real part')
    ax.set_title('Complex Array Visualization')

    # Add colorbar for imaginary part
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(colors)
    fig.colorbar(mappable, ax=ax, label='Imaginary part')

    plt.show()

# ============================================================================
# Perturbated Density Matrix Class
# ============================================================================
# Perturbated Density Matrix Class
class P_QOBJ: # Stands for Perturbated Qobj
    """ 
    P_QOBJ
    ===========================================
    A class to represent a density matrix and its perturbed versions.

    This is useful for estimating the QFIM using finite differences. 
    Each P_QOBJ object contains 2 density matrix QOBJ's for each parameter direction
    as well as an additional base density matrix. since there are 3 parameters for each
    channel for a total of 9 parameters there will be 19 density matrices stored in total. 

    This may seem computationally expensive but it is more efficient then symbolic
    differentiantion and allows for easy application of quantum gates to all perturbed states.

    Methods:
    ------

        configure_parameters(cls, parameters: list[list[float]], delta: float = 1e-6):
            Class method to configure the parameters and perturbation size for all P_QOBJ instances.

        __init__(self, base_rho: Qobj, location: int, entanglement_vector: list[int], qubit_id: int):
            Initialize the P_QOBJ object.

        set_state(self, new_state: "P_QOBJ"):
            Set the state of the P_QOBJ object to a new state. This is useful for updating the density matrices 
            of entangled qubits after applying gates.

        apply_gate(self, gate: Qobj, qubit_id: int, entanglement_vector: list[int]):
            Apply a quantum gate to all density matrices in the P_QOBJ object.

        generate_entanglement(self, qubit_id: int, entanglement_vector: list[int]):
            Generate entanglement between this qubit and a new qubit initialized in the |0> state.

        apply_error_channel(self, channel_number: int)):
            Apply a quantum error channel to all density matrices in the P_QOBJ object. Perturbations
            are applied in the X,Y, and Z directions for the specified channel.

    Attributes:
    -------------------

        channel_1 (dict[str: Qobj]): A dictionary containing the density matrices for channel 1.
        channel_2 (dict[str: Qobj]): A dictionary containing the density matrices for channel 2.
        channel_3 (dict[str: Qobj]): A dictionary containing the density matrices for channel 3.
        location (int): The location of the qubit in the circuit.
        entanglement_vector (list[int]): The entanglement vector of the qubit.
        qubit_id (int): The ID of the qubit in the circuit.

    """
    # Class variables (Circuit Variables)
    parameters: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)] # Default parameters for 3 qubits
    delta: float = 1e-6 # Default perturbation size

    # Set parameters for all density matrices in the system
    # These are class variables so they are shared across all
    # instances of the P_QOBJ class. I am new to OOP, if you have 
    # any questions on the syntax feel free to visit:

    # https://www.geeksforgeeks.org/python/python-oops-concepts/

    @classmethod
    def configure_parameters(cls, 
                             parameters: list[list[float]],
                             delta: float = 1e-6):
        cls.parameters=parameters
        cls.delta: float = delta

    def __init__(self, 
                 base_rho: Qobj,
                 location: int,
                 entanglement_vector: list[int],
                 qubit_id: int,
                 ):
        """ Initialize the P_QOBJ object.

        ARGS:
            base_rho (Qobj): The base density matrix.
            location (int): The location of the qubit in the circuit.
            entanglement_vector (list[int]): The entanglement vector of the qubit.
            qubit_id (int): The ID of the qubit in the circuit.
        """
        # Define the density matrices
        self._channel_1 = {
            'base': base_rho.copy(),
            'pX_plus': base_rho.copy(),
            'pX_minus': base_rho.copy(),
            'pY_plus': base_rho.copy(),
            'pY_minus': base_rho.copy(),
            'pZ_plus': base_rho.copy(),
            'pZ_minus': base_rho.copy()
        }
        self._channel_2 = {
            'base': base_rho.copy(),
            'pX_plus': base_rho.copy(),
            'pX_minus': base_rho.copy(),
            'pY_plus': base_rho.copy(),
            'pY_minus': base_rho.copy(),
            'pZ_plus': base_rho.copy(),
            'pZ_minus': base_rho.copy()
        }
        self._channel_3 = {
            'base': base_rho.copy(),
            'pX_plus': base_rho.copy(),
            'pX_minus': base_rho.copy(),
            'pY_plus': base_rho.copy(),
            'pY_minus': base_rho.copy(),
            'pZ_plus': base_rho.copy(),
            'pZ_minus': base_rho.copy()
        }

        # Store the qubit information
        self._location = location
        self._entanglement_vector = entanglement_vector
        self._qubit_id = qubit_id

    # ===========================================
    # Define GETTERS for each density matrix
    # ===========================================
    @property # The property decorator allows us to access the method as an attribute
    def channel_1(self):
        return self._channel_1
    
    @property
    def channel_2(self):
        return self._channel_2
    
    @property
    def channel_3(self):
        return self._channel_3
    
    @property
    def rho_dicts(self):
        return{
            'channel_1': self._channel_1,
            'channel_2': self._channel_2,
            'channel_3': self._channel_3
        }
    
    @property
    def location(self):
        return self._location
    
    @property
    def entanglement_vector(self):
        return self._entanglement_vector
    
    @property
    def qubit_id(self):
        return self._qubit_id

    # ==========================================================================
    # Define SETTERS for each density matrix
    # ==========================================================================
    @channel_1.setter
    def channel_1(self, new_c1: dict[str: Qobj]):
        self._channel_1 = new_c1

    @channel_2.setter
    def channel_2(self, new_c2: dict[str: Qobj]):
        self._channel_2 = new_c2

    @channel_3.setter
    def channel_3(self, new_c3: dict[str: Qobj]):
        self._channel_3 = new_c3

    
    def set_state(self, new_state: "P_QOBJ"): # Forward reference type hinting
        """ Set the state of the P_QOBJ object to a new state.

        ARGS:
            new_state (P_QOBJ): The new state to set the P_QOBJ object to.
        """
        self._channel_1 = new_state.channel_1
        self._channel_2 = new_state.channel_2
        self._channel_3 = new_state.channel_3

    @location.setter
    def location(self, new_location: int):
        self._location = new_location

    @entanglement_vector.setter
    def entanglement_vector(self, new_entanglement_vector: list[int]):
        self._entanglement_vector = new_entanglement_vector

    @qubit_id.setter
    def qubit_id(self, new_qubit_id: int):
        self._qubit_id = new_qubit_id

    # ========================================================================
    # Create methods to apply gates and error channels to all density matrices
    # ========================================================================

    # APPLY GATE METHOD
    # ==========================================
    def apply_gate(self, gate: Qobj):
        """ 
        apply_gate
        ===========================================
        Apply a quantum gate to all density matrices in the P_QOBJ object. 


        ARGS:
            gate (Qobj): The quantum gate to apply. This is represented as a 2 x 2
            density matrix object. That is then expanded to be applied to the full multi-qubit
            density matrix if necessary. If the qubit is not entangled with any other qubits
            then the gate is applied directly to the density matrix.
        """

        # Updated the perturbed density matrices by applying the gate to each one
        self.channel_1 = multi_qubit_apply_gate_dict(self.channel_1, gate, self.qubit_id, self.entanglement_vector)
        self.channel_2 = multi_qubit_apply_gate_dict(self.channel_2, gate, self.qubit_id, self.entanglement_vector)
        self.channel_3 = multi_qubit_apply_gate_dict(self.channel_3, gate, self.qubit_id, self.entanglement_vector)

    # Generate Entanglement Method
    # ==========================================
    def generate_entanglement(self):
        """ 
        generate_entanglement
        ===========================================
        Generate entanglement between the qubit represented by this P_QOBJ object
        and a new qubit in the |0> state.
        """

        # Update the perturbed density matrices by generating entanglement for each one
        self.channel_1 = generate_entanglement_multi_qubit_dict(self.channel_1, self.qubit_id, self.entanglement_vector)
        self.channel_2 = generate_entanglement_multi_qubit_dict(self.channel_2, self.qubit_id, self.entanglement_vector)
        self.channel_3 = generate_entanglement_multi_qubit_dict(self.channel_3, self.qubit_id, self.entanglement_vector)

    # APPLY ERROR METHOD
    # ==========================================
    def apply_error(self, channel_number: int):
        
        """ 
        appply_error
        ===========================================
        Apply the error channel to all density matrices in the P_QOBJ object. The
        perturbations are applied in the X, Y, and Z directions for the specified channel.

        ARGS:
            channel_number (int): The channel number to apply the error to. This is
            either 1, 2, or 3.
        """
        
        if channel_number == 1:
            # Only channel 1 applies perturbations in the X, Y, and Z directions
            self.channel_1 = multi_qubit_error_gate(self.channel_1, 
                                                    P_QOBJ.parameters[0], 
                                                    self.qubit_id, 
                                                    self.entanglement_vector, 
                                                    P_QOBJ.delta)
            self.channel_2 = multi_qubit_error_gate_non_perturbed(self.channel_2, 
                                                                  P_QOBJ.parameters[0], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)
            self.channel_3 = multi_qubit_error_gate_non_perturbed(self.channel_3, 
                                                                  P_QOBJ.parameters[0], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)
            

        elif channel_number == 2:
            self.channel_2 = multi_qubit_error_gate(self.channel_2, 
                                                    P_QOBJ.parameters[1], 
                                                    self.qubit_id, 
                                                    self.entanglement_vector, 
                                                    P_QOBJ.delta)
            self.channel_1 = multi_qubit_error_gate_non_perturbed(self.channel_1, 
                                                                  P_QOBJ.parameters[1], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)
            
            self.channel_3 = multi_qubit_error_gate_non_perturbed(self.channel_3, 
                                                                  P_QOBJ.parameters[1], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)

        elif channel_number == 3:
            self.channel_3 = multi_qubit_error_gate(self.channel_3, 
                                                    P_QOBJ.parameters[2], 
                                                    self.qubit_id, 
                                                    self.entanglement_vector, 
                                                    P_QOBJ.delta)
            
            self.channel_1 = multi_qubit_error_gate_non_perturbed(self.channel_1, 
                                                                  P_QOBJ.parameters[2], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)
            
            self.channel_2 = multi_qubit_error_gate_non_perturbed(self.channel_2, 
                                                                  P_QOBJ.parameters[2], 
                                                                  self.qubit_id, 
                                                                  self.entanglement_vector)
        else:
            raise ValueError("Channel number must be 1, 2, or 3.")
    

    # Calulate QFIM Method
    # ==========================================
    def calculate_QFIM(self) -> np.ndarray:
        """ 
        calculate_QFIM
        ===========================================
        Calculate the Quantum Fisher Information Matrix (QFIM) for the P_QOBJ object
        using finite differences.
        """


# ============================================================================


# ============================================================================
# Distribution Circuit Class
# ============================================================================
class DistributionCircuit:
    """ 
    DistributionCircuit
    ===========================================
    
    A class to represent a quantum distribution circuit.

    A Quantum distribution circuit object contains multiple P_QOBJ objects
    as well as global properties such as channel parameters and delta for perturbations.
    A full list of all operations applied to each qubit is also stored for debugging 
    and visualization purposes. 

    Methods:
    ------------------------------------------
        add_qubit(self, source_loc: int):
            Add a new qubit to the circuit at the specified location.

        apply_gate_to_qubit(self, qubit_id: int, gate: Qobj):
            Apply a quantum gate to a specific qubit in the circuit.

        update_entanglement_state(self, qubit_id: int):
            If one qubit has a gate applied to it, all entangled qubits must be updated as well.

        update_entanglement_vector(self, qubit_id: int, new_entanglement_vector: list[int]):
            Update the entanglement vector of a specific qubit in the circuit.

        generate_entanglement(self, qubit_id: int):
            Generate entanglement between a specific qubit and a new qubit initialized in the |0> state.

        move_qubit(self, qubit_id: int, end_loc: int):
            Move a specific qubit to a new location in the circuit. See details for movement rules.

        measure_all_qubits(self):
            Simulate the "Measure All Qubits" action by moving all qubits to end node 1 if they are in the center.

    Attributes:
    ------------------------------------------
        qubits (dict[int: P_QOBJ]): A dictionary containing the P_QOBJ objects for each qubit in the circuit.
            the key of each qubit is its unique integer ID also known as the qubit_id.
        action_history (list[list[int]]): A list to store action history for debugging and visualization purposes.
        num_qubits (int): The number of qubits in the circuit.
        channel_parameters (list[list[float]]): A list of channel parameters for each channel.
        delta (float): The perturbation size for finite difference calculations.
    """
    
    def __init__(self,
                 channel_parameters: list[list[float]],
                 delta: float = 1e-6):
        """ Initialize the DistributionCircuit object.

        ARGS:
            channel_parameters (list[list[float]]): A list of channel parameters for each channel.
            delta (float): The perturbation size for finite difference calculations.
        """
        self._qubits: dict[int: P_QOBJ] = {} # Dictionary to store P_QOBJ objects
        P_QOBJ.configure_parameters(channel_parameters, delta) # Configure class variables
        self._channel_parameters = channel_parameters
        self._delta = delta
        self._action_history: list[list[int]] = [] # List to store action history for debugging
        self._num_qubits: int = 0 # Number of qubits in the circuit
        self._gate_history: list[tuple[int]] = [] # List to store gate history for debugging
        self._state_graph: Graph = None # A Graph to store how the agent sees the state space

    # ==========================================
    # Define GETTERS and SETTERS
    # ==========================================

    @property
    def qubits(self):
        return self._qubits
    
    @qubits.setter
    def qubits(self, new_qubits: dict[int: P_QOBJ]):
        self._qubits = new_qubits

    @property
    def action_history(self):
        return self._action_history
    
    @action_history.setter
    def action_history(self, new_action_history: list[list[int]]):
        self._action_history = new_action_history

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @num_qubits.setter
    def num_qubits(self, new_num_qubits: int):
        self._num_qubits = new_num_qubits
    
    @property
    def channel_parameters(self):
        return self._channel_parameters
    
    @channel_parameters.setter
    def channel_parameters(self, new_channel_parameters: list[list[float]]):
        self._channel_parameters = new_channel_parameters
        P_QOBJ.configure_parameters(new_channel_parameters, self.delta)

    @property
    def delta(self):
        return self._delta
    
    @delta.setter
    def delta(self, new_delta: float):
        self._delta = new_delta
        P_QOBJ.configure_parameters(self.channel_parameters, new_delta)

    @property
    def gate_history(self):
        return self._gate_history
    
    @gate_history.setter
    def gate_history(self, new_gate_history: list[tuple[int]]):
        self._gate_history = new_gate_history


    @property
    def state_graph(self):
        return self._state_graph
    
    @state_graph.setter
    def state_graph(self, new_state_graph: "Graph"):
        self._state_graph = new_state_graph

    # ========================================================================
    # Create P_QOBJ getters for easy access to P_QOBJ attributes
    # ========================================================================

    def get_rho_dicts(self,qubit_id: int):
        """ 
        get_rho_dicts
        ===========================================
        Get the density matrix dictionaries for the P_QOBJ object.

        ARGS:
            qubit_id (int): The ID of the qubit in the circuit.
        RETURNS:
            dict[str: dict[str: Qobj]]: A dictionary containing the density matrix dictionaries for
            each channel.
        """
        return self.qubits[qubit_id].rho_dicts
    
    def get_location(self, qubit_id: int):
        """ 
        get_location
        ===========================================
        Get the location of the qubit in the circuit.

        ARGS:
            qubit_id (int): The ID of the qubit in the circuit.
        RETURNS:
            int: The location of the qubit in the circuit.
        """
        return self.qubits[qubit_id].location
    
    def get_entanglement_vector(self, qubit_id: int):
        """ 
        get_entanglement_vector
        ===========================================
        Get the entanglement vector of the qubit in the circuit.

        ARGS:
            qubit_id (int): The ID of the qubit in the circuit.
        RETURNS:
            list[int]: The entanglement vector of the qubit in the circuit.
        """
        return self.qubits[qubit_id].entanglement_vector

    # ============================================================================
    # Define methods to apply actions to the circuit
    # ============================================================================

    # Update action history method
    # ==========================================
    def update_action_history(self,
                              action: list[int]):
        """ 
        update_action_history
        ===========================================
        Update the action history of the circuit by appending a new action.
        ARGS:
            action (list[int]): The action to append to the action history.
        """

        self.action_history.append(action)

    # Add qubit method
    # ==========================================
    def add_qubit(self,
                  source_loc: int):
        """ Add a new qubit to the circuit at the specified location.

        ARGS:
            source_loc (int): The location to add the new qubit.
        """
        new_qubit_id = self.num_qubits
        new_rho = get_Z_basis_states()['RHO_ZERO']
        # The new qubit is not entangled with anyone
        new_p_qobj = P_QOBJ(new_rho, source_loc, [new_qubit_id], new_qubit_id)
        self.qubits[new_qubit_id] = new_p_qobj
        self.num_qubits += 1

    # Apply gate qubit method
    # ==========================================
    def apply_gate_to_qubit(self,
                            qubit_id: int,
                            gate_id: int):
        """ Apply a quantum gate to a specific qubit in the circuit.

        ARGS:
            qubit_id (int): The ID of the qubit to apply the gate to.
            gate (Qobj): The quantum gate to apply.
        """

        gates= get_all_gates()
        gates_list_keys = ['I', 'X', 'Y', 'Z', 'H', 'S', 'S_DAG']
        # This is the order in which gates are stored in the action vector
        if gate_id < 0 or gate_id >= len(gates_list_keys):
            raise ValueError(f"Invalid gate ID {gate_id}. Must be between 0 and {len(gates_list_keys)-1}.")

        gate_key = gates_list_keys[gate_id]
        gate = gates[gate_key] 

        self.qubits[qubit_id].apply_gate(gate)
        self.update_entanglement_state(qubit_id) # Update entangled qubits

        self.gate_history.append((qubit_id, gate_id)) # Update gate history for debugging

    # Update entanglement method
    # ==========================================
    def update_entanglement_state(self,
                             qubit_id: int):
        """ 
        update_entanglement_state
        ===========================================
        If one qubit has a gate applied to it, all entangled qubits must be updated as well.

        ARGS:
            qubit_id (int): The ID of the qubit to generate entanglement for.
        """
        
        entanglement_partners = self.qubits[qubit_id].entanglement_vector
        if len(entanglement_partners) > 1:
            for partner_id in entanglement_partners: # Go through each entanglement partner
                if partner_id != qubit_id: # Skip the original qubit
                    self.qubits[partner_id].set_state(self.qubits[qubit_id]) # Update the state to match the original qubit

    # Update entanglement vector method
    # ==========================================
    def update_entanglement_vector(self,
                                   qubit_id: int,
                                   new_qubit_id: int):
        """
        update_entanglement_vector
        ===========================================
        Update the entanglement vector of a qubit to include a new qubit. This is used
        when generating entanglement between a qubit and a new qubit. So that all entangled
        qubits can be properly updated when one qubit is modified.

        ARGS:
            qubit_id (int): The ID of the qubit to update.
            new_qubit_id (int): The ID of the new qubit to add to the entanglement vector.
        """

        # Add the new qubit to the entanglement vector of the original qubit
        entanglement_vector = self.qubits[qubit_id].entanglement_vector
        entanglement_vector.append(new_qubit_id)

        # Make sure all entangled qubits have the same entanglement vector
        for q_id in entanglement_vector:
            self.qubits[q_id].entanglement_vector = entanglement_vector
        
    # Generate entanglement method
    # ==========================================
    def generate_entanglement(self,
                              qubit_id: int):
        """ 
        generate_entanglement
        ===========================================
        Generate entanglement between a specific qubit and a new qubit initialized in the |0> state.

        ARGS:
            qubit_id (int): The ID of the qubit to generate entanglement for.
        """

        self.qubits[qubit_id].generate_entanglement() # Update the qubit's density matrices

        # Add the new qubit to the circuit
        new_qubit_id = self.num_qubits
        self.add_qubit(self.qubits[qubit_id].location) # New qubit at same location

        self.update_entanglement_vector(qubit_id, new_qubit_id) # Update entanglement vectors
        self.update_entanglement_state(qubit_id) 
        # Update entangled qubits to all have the same density matrices

    # Move qubit method
    # ==========================================
    def move_qubit(self,
                   qubit_id: int,
                   end_loc: int):
        """ 
        move_qubit
        ===========================================
        Move a specific qubit to a new location in the circuit. 

        Movement Cases:
        -------------------
            .1. The qubit is already at the desired end location not in center: 
               In this case the qubit is moved to the center and then back to the original location.

            .2. The qubit is at the center and the desired end location is not the center:
                In this case the qubit is moved to the desired end location.

            .3. The qubit is at an edge location and the desired end location is the center:
                In this case the qubit is moved to the center.

            .4. The qubit is in the center and the desired end location is also the center:
                In this case the qubit remains in the center.
    
            .5. The qubit is at an edge location and the desired end location is a different edge location:
                In this case the qubit is moved to the center and then to the desired end location.

        ARGS:
            qubit_id (int): The ID of the qubit to move.
            end_loc (int): The location to move the qubit to.
        """

        current_loc = self.qubits[qubit_id].location

        # Case 1: Qubit is already at desired end location not in center
        if current_loc == end_loc and current_loc != 0:
            # Move to center first
            self.qubits[qubit_id].apply_error(end_loc) # Apply channel 1 error
            self.qubits[qubit_id].location = 0 # Move to center
            self.update_entanglement_state(qubit_id) # Update entangled qubits

            # Then move back to original location
            self.qubits[qubit_id].apply_error(end_loc) # Apply channel 1 error
            self.qubits[qubit_id].location = end_loc # Move back to original location
            self.update_entanglement_state(qubit_id) # Update entangled qubits

        # Case 2: Qubit is at center and desired end location is not center
        elif current_loc == 0 and end_loc != 0:
            self.qubits[qubit_id].apply_error(end_loc) # Apply channel desired error
            self.qubits[qubit_id].location = end_loc # Move to desired end location
            self.update_entanglement_state(qubit_id) # Update entangled qubits

        # Case 3: Qubit is at edge location and desired end location is center
        elif current_loc != 0 and end_loc == 0:
            self.qubits[qubit_id].apply_error(current_loc) # Apply channel current error
            self.qubits[qubit_id].location = 0 # Move to center
            self.update_entanglement_state(qubit_id) # Update entangled qubits

        # Case 4: Qubit is in center and desired end location is also center
        elif current_loc == 0 and end_loc == 0:
            # No movement needed
            pass

        # Case 5: Qubit is at edge location and desired end location is different edge location
        elif current_loc != 0 and end_loc != 0 and current_loc != end_loc:
            # Move to center first
            self.qubits[qubit_id].apply_error(current_loc) # Apply channel current error
            self.qubits[qubit_id].location = 0 # Move to center
            self.update_entanglement_state(qubit_id) # Update entangled qubits

            # Then move to desired end location
            self.qubits[qubit_id].apply_error(end_loc) # Apply channel desired error
            self.qubits[qubit_id].location = end_loc # Move to desired end location
            self.update_entanglement_state(qubit_id) # Update entangled qubits

    # Measure all qubits method
    # ==========================================
    def measure_all_qubits(self):
        """ 
        measure_all_qubits
        ===========================================
        Simulate the "Measure All Qubits" action by moving all qubits to end node 1 if they are in the center.
        """

        for q_id in self.qubits:
            current_loc = self.qubits[q_id].location

            if current_loc == 0:
                # Move to end node 1
                self.qubits[q_id].apply_error(1) # Apply channel 1 error
                self.qubits[q_id].location = 1 # Move to end node 1
                self.update_entanglement_state(q_id) # Update entangled qubits

    # Output circuit information method
    # ==========================================
    def output_circuit_info(self):
        """ 
        output_circuit_info
        ===========================================
        Output the circuit information for debugging purposes.
        This function can be incredibly slow for large number of qubits, so use with caution.
        """

        print("Distribution Circuit Information:")
        print(f"Number of Qubits: {self.num_qubits}")
        print(f"Channel Parameters: {self.channel_parameters}")
        print(f"Delta: {self.delta}")
        print("Qubit States:")
        for q_id in self.qubits:
            p_qobj = self.qubits[q_id]
            print(f"Qubit ID: {q_id}")
            print(f" Location: {p_qobj.location}")
            print(f" Entanglement Vector: {p_qobj.entanglement_vector}")
            print(f" Channel 1 State: {p_qobj.channel_1['base']}")
            print(f" Channel 2 State: {p_qobj.channel_2['base']}")
            print(f" Channel 3 State: {p_qobj.channel_3['base']}")
            print("-----")

    # Output circuit density matrix method
    # ==========================================
    def output_circuit_density_matrices(self):
        """ 
        output_circuit_density_matrices
        ===========================================
        Output the density matrices of all qubits in the circuit for debugging purposes.
        """

        seen= {0}
        state=self.get_rho_dicts(0)['channel_1']['base'].full() # Start with qubit 0
        for q_id in self.qubits:
            if q_id not in seen:
                ent_vec= self.get_entanglement_vector(q_id)
                state= np.kron(state, self.get_rho_dicts(q_id)['channel_1']['base'].full())
                seen.update(ent_vec) # Add all entangled qubits to seen


        return state

    def plot_state(self):
        state= self.output_circuit_density_matrices()

        plot_complex_array(state)
    # Apply Error Without Moving Method
    # ==========================================
    def apply_error(self,
                    qubit_id: int,
                    channel_number: int):
        
        """ 
        apply_error
        ===========================================
        Apply the error channel to a specific qubit without moving it.
        ARGS:
            qubit_id (int): The ID of the qubit to apply the error to.
            channel_number (int): The channel number to apply the error to.
        """

        self.qubits[qubit_id].apply_error(channel_number)
        self.update_entanglement_state(qubit_id) # Update entangled qubits


    # ============================================================================
    # MAKE MOVE METHOD
    # ============================================================================
    def make_move(self,
                  action: list[int],
                  debug: bool = False):
        """ 
        Convert a list of action indices to a list of quantum gates.

        Action Encoding
        --------------------------
        The action for each move is one hot encoded as a vector of the following type:    
            [action_type, gate_id, qubit_id, end_loc, source_loc]

        There are 5 possible action types: 
            0: Apply Gate
            1: Move Qubit
            2: Add Qubit
            3: Generate Entanglement
            4: Measure All Qubits

        Each action type will use an action mask to only utilize the relevant parts of the action vector.
        0: Apply Gate
            gate_id: The gate to apply (0: I, 1: X, 2: Y, 3: Z, 4: H, 5: S 6: S_DAG)
            qubit_id: The qubit to apply the gate to
        1: Move Qubit
            qubit_id: The qubit to move
            end_loc: The location to move the qubit to (If the same as the current location the qubit is mvoed 
                    to the center and then back to the original location) If the qubit picks node 0 and is 
                    already there it will not move.
        2: Add Qubit
            source_loc: The location to add the qubit from.
        3: Generate Entanglement
            qubit_id: The qubit to entangle with a new qubit. This will be generated by a CNOT wherever the 
                    qubit currently is.
        4: Measure All Qubits
            No additional parameters needed. All qubits are measured. If not at an end node they are moved to 
            end node 1. 

            
        Thus the vector type is:
            0: action_type: either 0, 1, 2, 3, or 4
            1: gate_id: integer from 0 to 6
            2: qubit_id: integer from 0 to (num_qubits - 1)
            3: end_loc: integer from 0 to 3
            4: source_loc: integer from 1 to 3 (0 is not allowed as a source location)

        ARGS:
            action (list[int]): The action to apply to the circuit.
        RETURNS:
            None
        
        """

        action_type = action[0]
        gate_id = action[1]
        qubit_id = action[2]
        end_loc = action[3]
        source_loc = action[4]

        if action_type == 0: # Apply Gate
            gates= get_all_gates()
            gates_list_keys = ['I', 'X', 'Y', 'Z', 'H', 'S', 'S_DAG']
            # This is the order in which gates are stored in the action vector
            if gate_id < 0 or gate_id >= len(gates_list_keys):
                raise ValueError(f"Invalid gate ID {gate_id}. Must be between 0 and {len(gates_list_keys)-1}.")
            
            if debug:
                print(f"Applying gate {gates_list_keys[gate_id]} to qubit {qubit_id}.")

            self.apply_gate_to_qubit(qubit_id, gate_id)

        elif action_type == 1: # Move Qubit
            if debug:
                print(f"Moving qubit {qubit_id} to location {end_loc}.")
            self.move_qubit(qubit_id, end_loc)

        elif action_type == 2: # Add Qubit
            if debug:
                print(f"Adding new qubit at location {source_loc}.")
            self.add_qubit(source_loc)

        elif action_type == 3: # Generate Entanglement 
            if debug:
                print(f"Generating entanglement for qubit {qubit_id}.")
            self.generate_entanglement(qubit_id)

        elif action_type == 4: # Measure All Qubits
            if debug:
                print(f"Measuring all qubits.")
            self.measure_all_qubits()

        else:
            raise ValueError(f"Invalid action type {action_type}. Must be between 0 and 4.")
        
    
        self.update_action_history(action) # Update action history

    # ============================================================================
    # Calculate QFIM for all qubits method
    # ============================================================================

    def get_QFIM(self,
                 qubit_id: int) -> np.ndarray:
        """
        get_QFIM
        ===========================================
        Calculate the Quantum Fisher Information Matrix (QFIM) for a specific qubit in the circuit.
        ARGS:
            qubit_id (int): The ID of the qubit to calculate the QFIM for.
        RETURNS:
            np.ndarray: The QFIM for the specified qubit.
        """

        rho_dicts= self.get_rho_dicts(qubit_id)
        rho_dict_list= [rho_dicts['channel_1'], rho_dicts['channel_2'], rho_dicts['channel_3']]

        qfim= calculate_QFIM_direct_singular(rho_dict_list, self.delta)

        return qfim

    def get_all_QFIMs(self) -> np.ndarray:
        """
        get_all_QFIMs
        ===========================================
        Calculate the Quantum Fisher Information Matrix (QFIM) for all qubits in the circuit.
        RETURNS:
            The QFIM of the all qubits in the circuit.
        """

        qfim=np.zeros((9,9))
        seen= set() # To avoid double counting entangled qubits

        for qubit_id in self.qubits: # Iterate through all qubits
            if qubit_id not in seen: # If we haven't already calculated this qubit's entanglement group

                ent_vec= self.get_entanglement_vector(qubit_id)
                qfim_qubit= self.get_QFIM(qubit_id) # Get the QFIM for this qubit
                qfim += qfim_qubit # Add the QFIM to the total QFIM
                for q_id_i in ent_vec: # Iterate through entanglement vector
                    seen.add(q_id_i) # Mark this qubit as seen


        return qfim

    # ============================================================================
    # DIAGONALIZE SYSTEM METHOD
    # ============================================================================

    def diagonalize(self):
        """
        diagonalize
        ===========================================
        Diagonalize the density matrices of all qubits in the circuit. This will be used
        to enable the classical fisher information to be equal to the quantum fisher information.
        This is done by applying the conjugate transpose of all gates in reverse order.
        """
    
        reverse_gate_hist = self.gate_history[::-1] # Reverse the gate history for diagonalization

        for gate_info in reverse_gate_hist:
            qubit_id= gate_info[0]
            gate_id= gate_info[1]
            if gate_id == 5: # S gate
                gate_id= 6 # S_DAG gate
            elif gate_id == 6: # S_DAG gate
                gate_id= 5 # S gate
            self.apply_gate_to_qubit(qubit_id, gate_id) # Apply the conjugate transpose gate to the qubit
                
    # ===========================================================================
    # STATE REPRESENATION METHODS
    # ===========================================================================




#===========================================================================
# END OF DistributionCircuit CLASS
#===========================================================================

# ============================================================================
# NODE CLASS
# ============================================================================

class Node:
    """
    Node
    ===========================================

    A class to represent nodes in the state graph that the RLA will observe.
    Each node in the graph represents a qubit. And has distinct properties.

    Attributes:
    ------------------------------------------
        qubit_id (int): The unique identifier for the node.
        location (int): The location of the node in the network.
        qubit_state (Qobj): The quantum state of the qubit at this node.
        QFIM (np.ndarray): The Quantum Fisher Information Matrix for the qubit at this node.

    Notes:
    ------------------------------------------
        The qubit state for entangled qubits will be represented as the partial trace of 
        the state. Entanglement information will be stored in the edge weights.

        Currently each node contains only the base density matrix wihthout perturbations.
        In future versions we may want to store the perturbed density matrices as well.
    """

    def __init__(self,
                    qubit_id: int,
                    location: int,
                    qubit_state: Qobj,
                    QFIM: np.ndarray):
            """ Initialize the DistributionCircuit object.

            ARGS:
                channel_parameters (list[list[float]]): A list of channel parameters for each channel.
                delta (float): The perturbation size for finite difference calculations.
            """
            self._qubit_id = qubit_id
            self._location = location
            self._qubit_state = qubit_state
            self._QFIM = QFIM


    # ============================================================================
    # Define GETTERS and SETTERS
    # ============================================================================

    @property
    def qubit_id(self):
        return self._qubit_id
    
    @qubit_id.setter
    def qubit_id(self, new_qubit_id: int):
        self._qubit_id = new_qubit_id

    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, new_location: int):
        self._location = new_location

    @property
    def qubit_state(self):
        return self._qubit_state
    
    @qubit_state.setter
    def qubit_state(self, new_qubit_state: np.ndarray):
        self._qubit_state = new_qubit_state

    @property
    def QFIM(self):
        return self._QFIM
    
    @QFIM.setter
    def QFIM(self, new_QFIM: np.ndarray):
        self._QFIM = new_QFIM


# ============================================================================
# EDGE CLASS
# ============================================================================

class Edge:
    """
    Edge
    ===========================================

    A class to represent edges in the state graph that the RLA will observe.
    Each edge in the graph represents a connection between two nodes (qubits).

    Attributes:
    ------------------------------------------
        connection (tuple[int, int]): A tuple representing the connection between two nodes (from_node, to_node).
             Since this is an undirected graph, the order of the nodes does not matter. But by convention we store it as (from_node, to_node) where from_node > to_node.

        entanglement entropy (float): The entanglement entropy between the two nodes connected by this edge.
    
    # Notes:
    ------------------------------------------
        The entanglement entropy is a measure of the quantum entanglement between the two qubits. Since entanglement is symmetric, the entanglement entropy is the same regardless of the direction of the edge.

        All subgraphs of our graph will be fully connected since entanglement is shared among all qubits in the entanglement vector. This representation of the state allows the agent to see the full entanglement structure of the circuit, as 
        well as how 2 qubits are entangled with each other.

    
    """

    def __init__(self,
                 connection: tuple[int, int],
                 ent_entropy: float):
        """ Initialize the Edge object.

        ARGS:
            connection (tuple[int, int]): A tuple representing the connection between two nodes (from_node, to_node).
            weight (float): The entanglement entropy between the two nodes connected by this edge.
        """

        if connection[0] < connection[1]:
            connection= (connection[1], connection[0]) # Ensure from_node > to_node for consistency

        self._from_node = connection[0]
        self._to_node = connection[1]
        self._connection = connection
        self._ent_entropy = ent_entropy


    # Define Getters and Setters
    #---------------------------------------------------------------------------

    @property
    def connection(self):
        return self._connection
    
    @connection.setter
    def connection(self, new_connection: tuple[int, int]):
        self._connection = new_connection

    @property
    def from_node(self):
        return self._from_node
    
    @from_node.setter
    def from_node(self, new_from_node: int):
        self._from_node = new_from_node

    @property
    def to_node(self):
        return self._to_node
    
    @to_node.setter
    def to_node(self, new_to_node: int):
        self._to_node = new_to_node

    @property
    def ent_entropy(self):
        return self._ent_entropy
    
    @ent_entropy.setter
    def ent_entropy(self, new_ent_entropy: float):
        self._ent_entropy = new_ent_entropy

    

# ============================================================================
# GRAPH CLASS
# ============================================================================

class Graph:
    """
    Graph
    ===========================================

    A class to represent the state graph that the RLA will observe.
    The graph contains nodes and edges representing qubits and their entanglement.

    Attributes:
    ------------------------------------------
        node_list (list[Node]): A list of Node objects representing the qubits in the circuit.
        edge_list (list[Edge]): A list of Edge objects representing the entanglement between
    """

    def __init__(self,
                 params: list[list[float]],
                 node_dict: dict[int: [Node]] = {},
                 edge_list: list[Edge] = [],
                 ):
        """ Initialize the Graph object.
        """
        self._nodes: dict[int:[Node]] = node_dict # Dictionary to store Node objects
        self._edges: list[Edge] = edge_list # Dictionary to store Edge objects
        self._total_QFIM: np.ndarray = np.zeros((9,9)) # Total QFIM for the graph

        self._params: list[list[float]] = params # Channel parameters for the graph
    

    # Define Getters and Setters
    #---------------------------------------------------------------------------

    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, new_nodes: dict[int: Node]):
        self._nodes = new_nodes

    @property
    def edges(self):
        return self._edges
    
    @edges.setter
    def edges(self, new_edges: dict[tuple[int, int]: Edge]):
        self._edges = new_edges

    @property
    def total_QFIM(self):
        return self._total_QFIM
    
    @total_QFIM.setter
    def total_QFIM(self, new_total_QFIM: np.ndarray):
        self._total_QFIM = new_total_QFIM

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, new_params: list[list[float]]):
        self._params = new_params


    # ============================================================================
    # ADD NODE AND EDGE METHODS
    # ============================================================================
    def add_node(self,
                 node: Node):
        """ 
        add_node
        ===========================================
        Add a new node to the graph. 
        ARGS:
            node (Node): The Node object to add to the graph.
        """
        self.nodes[node.qubit_id] = node

    def add_edge(self,
                 edge: Edge):
        """ 
        add_edge
        ===========================================
        Add a new edge to the graph.

        ARGS:
            edge (Edge): The Edge object to add to the graph.
        """

        self.edge_list.append(edge)

    # ============================================================================
    # UPDATE GRAPH METHODS
    # ============================================================================

    def make_new_node(self,
                      qubit_id: int,
                      location: int):
        
        """
        make_new_node
        ===========================================

        Create a new Node object and add it to the graph.

        ARGS:
            qubit_id (int): The ID of the qubit to create the node for.
            location (int): The location of the qubit in the network.
        """

        state: Qobj =get_Z_basis_states()['RHO_ZERO']
        # Qubit initialized in |0> state

        qfim= np.zeros((9,9)) # Empty QFIM for new node
        new_node= Node(qubit_id, location, state, qfim) # Create a new Node object with empty state and QFIM

        # Add the new node to the graph
        self.nodes[qubit_id]= new_node
        
    def make_new_node_entangled(self,
                                qubit_id: int,
                                location: int,
                                rho: Qobj,
                                QFIM: np.ndarray,
                                ent_vec: list[int]):
        """
        make_new_node_entangled
        =========================================== 
        Create a new Node object for an entangled qubit and add it to the graph.
        ARGS:
            qubit_id (int): The ID of the qubit to create the node for.
            location (int): The location of the qubit in the network.
            rho (Qobj): The quantum state of the qubit.
            QFIM (np.ndarray): The Quantum Fisher Information Matrix for the qubit.
        """
        index= ent_vec.index(qubit_id)
        state=rho.ptrace(index) # Get the partial trace for entangled qubits
        new_node= Node(qubit_id, location, state, QFIM) # Create a new Node object

        # Add the new node to the graph
        self.nodes[qubit_id]= new_node

    def update_node(self,
                     qubit_id: int,
                     location: int,
                     qubit_state: Qobj,
                     QFIM: np.ndarray,
                     ent_vec: list[int]):
        """
        update_node
        ===========================================
        Update the properties of a node in the graph.

        ARGS:
            qubit_id (int): The ID of the qubit to update.
            location (int): The new location of the qubit in the network.
            qubit_state (Qobj): The new quantum state of the qubit.
            QFIM (np.ndarray): The new Quantum Fisher Information Matrix for the qubit.
        """
        # If the qubit is entangled with others, update their states as well
        if len(ent_vec) > 1:
            for q_id in ent_vec:
                    other_index= ent_vec.index(q_id)
                    other_state= qubit_state.ptrace(other_index)
                    other_node: Node = self.nodes[q_id]
                    other_node.qubit_state= other_state # Update the state of the other entangled qubits


        node: Node = self.nodes[qubit_id]
        node.location= location
        node.qubit_state= qubit_state
        node.QFIM= QFIM

    def update_edges(self,
                     ent_vec: list[int],
                     new_qubit_id: int,
                     rho: Qobj):
        """
        update_edges
        ===========================================
        Update the edges in the graph based on the entanglement vector.
        ARGS:
            ent_vec (list[int]): The entanglement vector to update the edges for.
            new_qubit_id (int): The ID of the new qubit to add edges for.
            rho (Qobj): The density matrix of the new qubit.
        """
        # Create edges between all nodes with the new qubit
        for q_id in ent_vec:
            if q_id != new_qubit_id:
                connection= (q_id, new_qubit_id)

                # Gets the qubit states for entanglement entropy calculation
                
                
                ent_entropy= calculate_entanglement_entropy(rho,new_qubit_id,ent_vec)
                new_edge= Edge(connection, ent_entropy)
                self.add_edge(new_edge)





# ============================================================================
# Testing Functions
# ============================================================================
def get_test_circuit():
    """
    This function is used to generate a test DistributionCircuit for debugging purposes. 
    Generate a test circuit with 4 qubits:
        Qubit 0: |0> state at node 1
        Qubit 1: |+> state at node 2
        Qubit 2 and 3: Entangled pair at node 3. Both in the |0> state initially.
    RETURNS:
        DistributionCircuit: 
        A DistributionCircuit object containing the quantum circuit information.
    """
    test_circuit = DistributionCircuit(channel_parameters=[[0.01, 0.02, 0.03],
                                                          [0.04, 0.05, 0.06],
                                                          [0.07, 0.08, 0.09]],
                                       delta=1e-6)
    # Add qubit 0
    test_circuit.add_qubit(1) # Node 1 # by default in |0> state
    # Add qubit 1
    test_circuit.add_qubit(2) # Node 2
    test_circuit.apply_gate_to_qubit(1, 4) # Apply H gate to qubit 1 to create |+> state
    # Add qubit 2
    test_circuit.add_qubit(3) # Node 3 # This is qubit 2
    test_circuit.generate_entanglement(2) # Generate entanglement between qubit 2 and new qubit 3

    return test_circuit

def action_dict2dc(action_dict: dict[int: [dict[str: Qobj],list[int],int]],
                   parameters: list[list[float]],
                   delta: float=1e-6) -> DistributionCircuit:
    
    dc= DistributionCircuit(channel_parameters=parameters, delta=delta)

     # Populate the DistributionCircuit with P_QOBJ objects from the action_dict
    for qubit_id in action_dict:
        rho= action_dict[qubit_id][0]['base']
        loc= action_dict[qubit_id][2]
        ent_vec= action_dict[qubit_id][1]
        p_qobj= P_QOBJ(rho, loc, ent_vec, qubit_id)
        dc.qubits[qubit_id]= p_qobj
        dc.num_qubits +=1

    return dc

def get_random_dc(params,delta):
    dc= DistributionCircuit(channel_parameters=params, delta=delta)
    dc.add_qubit(random.randint(1,3)) # Add first qubit at random source location
    moves= random.randint(5,15) # Random number of moves between 5 and 15

    for _ in range(moves):
        action=[random.randint(0,4), # action type
                random.randint(0,6), # gate id
                random.randint(0, dc.num_qubits -1),
                random.randint(0,3), # end loc
                random.randint(1,3)] # source loc 
        
        #  Thus the vector type is:
        #     0: action_type: either 0, 1, 2, 3, or 4
        #     1: gate_id: integer from 0 to 6
        #     2: qubit_id: integer from 0 to (num_qubits - 1)
        #     3: end_loc: integer from 0 to 3
        #     4: source_loc: integer from 1 to 3 (0 is not allowed as a source location)
        
        dc.make_move(action)

    print(f"Generated random circuit with {dc.num_qubits} qubits and {moves} moves.")
    return dc


@profile
def gauge_efficiency():
    """
    gauge_efficiency
    ===========================================
    Test the efficiency of the gauge function for calculating the QFIM of a random DistributionCircuit
    
    To run this funciton with profiling, call the function in the main block and use the following command:
        kernprof -l -v helper_functions/circuit_builder_1.py

    """
    params= [[0.01, 0.02, 0.03],
             [0.04, 0.05, 0.06],
             [0.07, 0.08, 0.09]]
    
    delta = 1e-6

    for i in range(10):
        dc=get_random_dc(params,delta)
        qfim= dc.get_all_QFIMs()
        dc.diagonalize()
        state= dc.output_circuit_density_matrices()


if __name__ == "__main__":
    gauge_efficiency()




"""
A script containing unit tests for the circuit_builder_1 module. To conduct these
tests at any times enter pytest tests/test_circuit_builder_1.py in the terminal.
"""
# ============================================================================
# IMPORTS
# ============================================================================
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import sympy as sp
import numpy as np
from helper_functions import qubit_mover_2 as qm2
from helper_functions import circuit_builder_1 as cb1

from qutip import Qobj
import qutip as qt

# ============================================================================
# Test get functions and custom quantum operations
# ============================================================================


def test_get_all_gates():
    gates = cb1.get_all_gates()
    expected_keys = {'I', 'X', 'Y', 'Z', 'H', 'S', 'S_DAG','CNOT'}
    assert set(gates.keys()) == expected_keys
    for gate in gates.values():
        assert isinstance(gate, Qobj)

def test_get_all_basis_states():
    states = cb1.get_all_basis_states()
    expected_keys = {'RHO_ZERO', 'RHO_ONE', 'RHO_PLUS', 'RHO_MINUS', 'RHO_R', 'RHO_L'}
    assert set(states.keys()) == expected_keys
    for state in states.values():
        assert isinstance(state, Qobj)

def test_get_Z_basis_states():
    states = cb1.get_Z_basis_states()
    expected_keys = {'RHO_ZERO', 'RHO_ONE'}
    assert set(states.keys()) == expected_keys
    for state in states.values():
        assert isinstance(state, Qobj)

def test_get_X_basis_states():
    states = cb1.get_X_basis_states()
    expected_keys = {'RHO_PLUS', 'RHO_MINUS'}
    assert set(states.keys()) == expected_keys
    for state in states.values():
        assert isinstance(state, Qobj)

def test_get_Y_basis_states():
    states = cb1.get_Y_basis_states()
    expected_keys = {'RHO_R', 'RHO_L'}
    assert set(states.keys()) == expected_keys
    for state in states.values():
        assert isinstance(state, Qobj)

def test_get_perturbed_0_state():

    ket0=cb1.get_Z_basis_states()['RHO_ZERO']
    perturbed_state = cb1.get_perturbed_0_state()
    keys= perturbed_state.keys()
    assert keys == {'base','pZ_plus','pZ_minus','pX_plus','pX_minus','pY_plus','pY_minus'}
    assert perturbed_state['base'] == perturbed_state['pZ_plus']
    assert perturbed_state['base'] == ket0

# ============================================================================
# Test apply gate function
# ============================================================================

def test_apply_gate_X():
    rho_zero = cb1.get_all_basis_states()['RHO_ZERO']
    X_gate = cb1.get_all_gates()['X']
    result = cb1.apply_gate(rho_zero, X_gate)
    expected = cb1.get_all_basis_states()['RHO_ONE'] # |1><1|
    assert result == expected

def test_apply_gate_X_plus():
    rho_plus = cb1.get_all_basis_states()['RHO_PLUS']
    X_gate = cb1.get_all_gates()['X']
    result = cb1.apply_gate(rho_plus, X_gate)
    expected = rho_plus # |+><+| No change
    assert result == expected

def test_apply_gate_H():
    rho_zero = cb1.get_all_basis_states()['RHO_ZERO']
    H_gate = cb1.get_all_gates()['H']
    result = cb1.apply_gate(rho_zero, H_gate)
    expected = cb1.get_all_basis_states()['RHO_PLUS'] # |+><+|
    assert result == expected

def test_apply_gate_H_minus():
    rho_minus = cb1.get_all_basis_states()['RHO_MINUS']
    H_gate = cb1.get_all_gates()['H']
    result = cb1.apply_gate(rho_minus, H_gate)
    expected = cb1.get_all_basis_states()['RHO_ONE'] # |1><1|
    assert result == expected

def test_apply_gate_S():
    rho_plus = cb1.get_all_basis_states()['RHO_PLUS']
    S_gate = cb1.get_all_gates()['S']
    result = cb1.apply_gate(rho_plus, S_gate)
    expected = cb1.get_all_basis_states()['RHO_R'] # |R><R|
    assert result == expected

def test_apply_gate_S_R():
    rho_r = cb1.get_all_basis_states()['RHO_R']
    S_gate = cb1.get_all_gates()['S']
    result = cb1.apply_gate(rho_r, S_gate)
    expected = cb1.get_all_basis_states()['RHO_MINUS'] # |-><-|
    assert result == expected

def test_apply_gate_Sdag():
    rho_r = cb1.get_all_basis_states()['RHO_R']
    Sdag_gate = cb1.get_all_gates()['S'].dag()
    result = cb1.apply_gate(rho_r, Sdag_gate)
    expected = cb1.get_all_basis_states()['RHO_PLUS'] # |+><+|
    assert result == expected

def test_apply_gate_SDAG():
    rho_minus = cb1.get_all_basis_states()['RHO_MINUS']
    Sdag_gate = cb1.get_all_gates()['S'].dag()
    expect = cb1.apply_gate(rho_minus, Sdag_gate)
    
    SDAG_gate= cb1.get_all_gates()['S_DAG']
    result = cb1.apply_gate(rho_minus, SDAG_gate)

    assert result == expect
    

# ============================================================================
# APPLY ERROR FUNCTIONS
# ============================================================================

def test_error_gate_bit_flip():
    rho_zero = cb1.get_all_basis_states()['RHO_ZERO']
    p = [0.5,0,0] # 50% bit flip
    result = cb1.error_gate(rho_zero, p)
    expected = Qobj([[0.5,0],[0,0.5]]) # 50% |0><0| + 50% |1><1|
    assert result == expected

def test_error_gate_phase_flip():
    rho_zero = cb1.get_all_basis_states()['RHO_ZERO']
    p = [0,0,0.5] # 50% phase flip
    result = cb1.error_gate(rho_zero, p)
    expected = rho_zero # No change for |0><0|
    assert result == expected

def test_error_gate_all_flips():
    rho_plus = cb1.get_all_basis_states()['RHO_PLUS']
    p = [0.5,0.5,0.5] # 50% bit flip and 50% phase flip
    result = cb1.error_gate(rho_plus, p)
    expected = Qobj([[0.5,-0.5],[-0.5,0.5]]) # 50% |0><0| + 50% |1><1|
    assert result == expected


# ============================================================================
# TEST ERROR GATE PERTURBATION FUNCTION
# ============================================================================

def test_error_gate_perturbed():
    rho_dict= cb1.get_perturbed_0_state()

    for _ in range(10):
        parameters = [np.random.uniform(0,0.3) for _ in range(3)]
        delta= 1e-6
        perturbed_ps = cb1.error_gate_perturbed(rho_dict,parameters, h=delta)
        expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
        assert set(perturbed_ps.keys()) == expected_keys

        expected_base = cb1.error_gate(rho_dict['base'], parameters)

        assert perturbed_ps['base'] == expected_base

        # Check that perturbations are correctly applied
        for key in expected_keys:
            if key != 'base':
                perturbed_state = perturbed_ps[key]
                
                # Ensrure we are perturbing the correct parameter and that the perturbation is correct

                if 'X' in key:
                    plus_state= cb1.error_gate(rho_dict['base'], [parameters[0] + delta, parameters[1], parameters[2]])
                    minus_state= cb1.error_gate(rho_dict['base'], [parameters[0] - delta, parameters[1], parameters[2]])
                    
                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)

                    
                elif 'Y' in key:
                    plus_state= cb1.error_gate(rho_dict['base'], [parameters[0], parameters[1]+delta, parameters[2]])
                    minus_state= cb1.error_gate(rho_dict['base'], [parameters[0], parameters[1]-delta, parameters[2]])

                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    
                    
                elif 'Z' in key:
                    plus_state= cb1.error_gate(rho_dict['base'], [parameters[0], parameters[1], parameters[2]+delta])
                    minus_state= cb1.error_gate(rho_dict['base'], [parameters[0], parameters[1], parameters[2]-delta])
                    
                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)

                
def test_error_gate_perturbed_double():
    rho_dict= cb1.get_perturbed_0_state()
    
    for _ in range(10):
        parameters_1 = [np.random.uniform(0,0.3) for _ in range(3)] 
        parameters_2 = [np.random.uniform(0,0.3) for _ in range(3)]
        delta= 1e-6

        perturbed_ps_1 = cb1.error_gate_perturbed(rho_dict,parameters_1, h=delta)
        perturbed_ps_2 = cb1.error_gate_perturbed(perturbed_ps_1,parameters_2, h=delta)


        expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
        assert set(perturbed_ps_2.keys()) == expected_keys

        # Check that the base state is correct
        expected_base = cb1.error_gate(cb1.error_gate(rho_dict['base'], parameters_1), parameters_2)
        assert perturbed_ps_2['base'] == expected_base

        # Check that perturbations are correctly applied
        for key in expected_keys:
            if key != 'base':
                perturbed_state = perturbed_ps_2[key]
                
                # Ensrure we are perturbing the correct parameter and that the perturbation is correct

                if 'X' in key:
                    plus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0]+delta,
                                                              parameters_1[1],
                                                              parameters_1[2]]
                                                              ), 
                                                [parameters_2[0] + delta, 
                                                 parameters_2[1], 
                                                 parameters_2[2]]
                                                 )
                    
                    minus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0]-delta,
                                                              parameters_1[1],
                                                              parameters_1[2]]
                                                              ), 
                                                [parameters_2[0] - delta, 
                                                 parameters_2[1], 
                                                 parameters_2[2]]
                                                 )
                    
                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)

                    
                elif 'Y' in key:
                    plus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0],
                                                              parameters_1[1]+delta,
                                                              parameters_1[2]]
                                                              ), 
                                                [parameters_2[0], 
                                                 parameters_2[1]+delta, 
                                                 parameters_2[2]]
                                                 )
                    
                    minus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0],
                                                              parameters_1[1]-delta,
                                                              parameters_1[2]]
                                                              ), 
                                                [parameters_2[0], 
                                                 parameters_2[1]-delta, 
                                                 parameters_2[2]]
                                                 )

                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    
                    
                elif 'Z' in key:
                    plus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0],
                                                              parameters_1[1],
                                                              parameters_1[2]+delta]
                                                              ), 
                                                [parameters_2[0], 
                                                 parameters_2[1], 
                                                 parameters_2[2]+delta]
                                                 )
                    
                    minus_state= cb1.error_gate(cb1.error_gate(rho_dict['base'], 
                                                              [parameters_1[0],
                                                              parameters_1[1],
                                                              parameters_1[2]-delta]
                                                              ), 
                                                [parameters_2[0], 
                                                 parameters_2[1], 
                                                 parameters_2[2]-delta]
                                                 )
                    
                    if 'plus' in key:
                        np.testing.assert_allclose(plus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)
                    else:
                        np.testing.assert_allclose(minus_state.full(),(perturbed_state.full()),rtol=1e-6,atol=1e-8)


# ============================================================================
# TEST APPLY GATE RHO DICT FUNCTION
# ============================================================================

def test_apply_gate_rho_dict():
    rho_dict= cb1.get_perturbed_0_state()
    gate= cb1.get_all_gates()['H']

    applied_rho_dict= cb1.apply_gate_rho_dict(rho_dict, gate)

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(applied_rho_dict.keys()) == expected_keys

    expected_base = cb1.apply_gate(rho_dict['base'], gate)
    assert applied_rho_dict['base'] == expected_base

    for key in expected_keys:
        if key != 'base':
            perturbed_state = applied_rho_dict[key]
            
            # Ensure the gate is applied correctly to each perturbed state
            expected_state = cb1.apply_gate(rho_dict[key], gate)
            assert perturbed_state == expected_state


def test_apply_gate_rho_dict_error():
    rho_dict= cb1.get_perturbed_0_state()
    gate= cb1.get_all_gates()['X']  # Bit-flip gate

    ket0= cb1.get_Z_basis_states()['RHO_ZERO']

    for _ in range(10):

        parameters= [np.random.uniform(0,0.3) for _ in range(3)]  # Example error probabilities
        error_rho_dict = cb1.error_gate_perturbed(rho_dict, parameters, h=1e-6)

        applied_rho_dict= cb1.apply_gate_rho_dict(error_rho_dict, gate)

        expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
        assert set(applied_rho_dict.keys()) == expected_keys

        expected_base = cb1.apply_gate(cb1.error_gate(ket0, parameters), gate)

        np.testing.assert_allclose(applied_rho_dict['base'].full(), expected_base.full(), rtol=1e-6, atol=1e-8)

        parameters_pX_plus = parameters.copy()
        parameters_pX_plus[0] += 1e-6
        expected_base_pX = cb1.apply_gate(cb1.error_gate(ket0, parameters_pX_plus), gate)
        np.testing.assert_allclose(applied_rho_dict['pX_plus'].full(), expected_base_pX.full(), rtol=1e-6, atol=1e-8)


        parameters_pZ_plus = parameters.copy()
        parameters_pZ_plus[2] += 1e-6
        expected_base_pZ = cb1.apply_gate(cb1.error_gate(ket0, parameters_pZ_plus), gate)
        np.testing.assert_allclose(applied_rho_dict['pZ_plus'].full(), expected_base_pZ.full(), rtol=1e-6, atol=1e-8)


# ============================================================================
# TEST GENERATE ENTANGLEMENT FUNCTION
# ============================================================================

def test_generate_entanglement_rho_dict_sup():

    H_gate= cb1.get_all_gates()['H']
    # Generate bell state
    rho_dict= cb1.get_perturbed_0_state()

    sup_rho_dict = cb1.apply_gate_rho_dict(rho_dict, H_gate)

    # Generate entanglement
    entangled_rho_dict= cb1.generate_entanglement_rho_dict(sup_rho_dict)
    
    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(entangled_rho_dict.keys()) == expected_keys

    # Check base state is Bell state |Φ+><Φ+|
    expected_bell_state = Qobj([[0.5, 0, 0, 0.5],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0.5, 0, 0, 0.5]])
    np.testing.assert_allclose(entangled_rho_dict['base'].full(), expected_bell_state.full(), rtol=1e-6, atol=1e-8)

def test_generate_entanglement_rho_dict_zero():
    
    # Generate bell state
    rho_dict= cb1.get_perturbed_0_state()

    entangled_rho_dict= cb1.generate_entanglement_rho_dict(rho_dict)
    
    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(entangled_rho_dict.keys()) == expected_keys

    # Check base state is Bell state |00><00|
    expected_bell_state = Qobj([[1, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    np.testing.assert_allclose(entangled_rho_dict['base'].full(), expected_bell_state.full(), rtol=1e-6, atol=1e-8)


# ============================================================================
# TEST GENERATE ENTANGLEMENT MULTI QUBIT
# ============================================================================

def test_generate_entanglement_multi_qubit():
    # Initialize three qubits in |0><0| state
    q0=cb1.get_Z_basis_states()['RHO_ZERO']
    q1=cb1.get_Z_basis_states()['RHO_ONE']
    q2=cb1.get_Z_basis_states()['RHO_ZERO']

    total_state= qt.tensor(q0,q1,q2)

    entanglement_vector = [0,1,2] # Entanglement vector for q0
    entangled_state= cb1.generate_entanglement_multi_qubit(total_state, 1 ,entanglement_vector)

    # Expected state is |0101><0101|
    
    ket_0 = cb1.get_Z_basis_states()['RHO_ZERO']
    ket_1 = cb1.get_Z_basis_states()['RHO_ONE']

    expected_state = qt.tensor(ket_0, ket_1, ket_0, ket_1)

    assert entangled_state == expected_state

def test_generate_entanglement_multi_qubit_sup():
    q0=cb1.get_X_basis_states()['RHO_PLUS']
    q1=cb1.get_Z_basis_states()['RHO_ZERO']
    

    total_state= qt.tensor(q0,q1) # This is not an entangled state

    entanglement_vector = [0,1] # Not really an entanglement vector for q0
    entangled_state= cb1.generate_entanglement_multi_qubit(total_state, 0 ,entanglement_vector)

    # Expected state is: |+00+><+00+|

    ket_0 = qt.basis(2,0)
    ket_1 = qt.basis(2,1)

    # expected_state = (|000> + |101> )1/sqrt(2)

    
    expected_state = (qt.tensor(ket_0,ket_0,ket_0) + qt.tensor(ket_1,ket_0,ket_1)).proj()
    expected_state = (expected_state).unit()
    
    assert entangled_state == expected_state

def test_generate_entanglement_multi_qubit_complex():
    # Set up the state: 
    cnot= cb1.get_all_gates()['CNOT']
    q0=cb1.get_X_basis_states()['RHO_PLUS']
    q1=cb1.get_Z_basis_states()['RHO_ZERO']

    total_state= qt.tensor(q0,q1) # This is not an entangled state

    bell_state = cnot * total_state * cnot.dag()

    entanglement_vector = [0,1] # Entanglement vector for q0
    entangled_state= cb1.generate_entanglement_multi_qubit(bell_state, 0 ,entanglement_vector)

    # Should be |+++><+++| = (|000> + |111>)(<000| + <111|) /2
    ket_0 = qt.basis(2,0)
    ket_1 = qt.basis(2,1)

    expected_state = (qt.tensor(ket_0,ket_0,ket_0) + qt.tensor(ket_1,ket_1,ket_1)).proj() /2
    

    assert entangled_state == expected_state

def test_generate_entanglement_multi_qubit_single_case():
    # Initialize single qubit in |0><0| state
    q10=cb1.get_Z_basis_states()['RHO_ZERO']

    total_state= qt.tensor(q10)

    entanglement_vector = [10] # Entanglement vector for q10
    entangled_state= cb1.generate_entanglement_multi_qubit(total_state, 10 ,entanglement_vector)

    # Expected state is |00><00|
    
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']
    cnot= cb1.get_all_gates()['CNOT']
    expected_state = cnot * qt.tensor(rho_0, rho_0) * cnot.dag()


    assert entangled_state == expected_state


# ============================================================================
# TEST MULTI QUBIT APPLY GATE
# ===========================================================================

def test_multi_qubit_apply_gate():
    # Initialize three qubits in |0><0| state
    q0=cb1.get_Z_basis_states()['RHO_ZERO']
    q1=cb1.get_Z_basis_states()['RHO_ZERO']
    q2=cb1.get_Z_basis_states()['RHO_ZERO']

    total_state= qt.tensor(q0,q1,q2)

    entanglement_vector = [0,1,2] # Not really an entanglement vector for q0

    X_gate= cb1.get_all_gates()['X']

    new_state= cb1.multi_qubit_apply_gate(total_state, X_gate, 1, entanglement_vector)

    # Expected state is |010><010| # not really this state as the state is mixed.
    
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']
    rho_1 = cb1.get_Z_basis_states()['RHO_ONE']

    expected_state = qt.tensor(rho_0, rho_1, rho_0)

    assert new_state == expected_state

def test_multi_qubit_apply_gate_entangled():
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']

    rho_combined=cb1.generate_entanglement_multi_qubit(rho_0,2,[2])

    rho_combined_2=cb1.generate_entanglement_multi_qubit(rho_combined,2,[2,5])

    H_gate= cb1.get_all_gates()['H']

    new_state= cb1.multi_qubit_apply_gate(rho_combined_2, H_gate, 5, [2,5,7])

    # Expected state is |0+0><0+0|

    ket_zero= qt.basis(2,0)
    ket_plus= (qt.basis(2,0) + qt.basis(2,1)).unit()

    expected_state = qt.tensor(ket_zero, ket_plus, ket_zero).proj()

    assert new_state == expected_state

def test_multi_qubit_apply_gate_single_qubit():
    # Initialize single qubit in |0><0| state
    q10=cb1.get_Z_basis_states()['RHO_ZERO']

    total_state= qt.tensor(q10)

    entanglement_vector = [10] # Entanglement vector for q10

    X_gate= cb1.get_all_gates()['X']

    new_state= cb1.multi_qubit_apply_gate(total_state, X_gate, 10, entanglement_vector)

    # Expected state is |1><1|
    
    


    assert new_state == cb1.get_Z_basis_states()['RHO_ONE']

# ===========================================================================
# TEST ERROR GATE TENSOR FUNCTION
# ===========================================================================

def test_error_gate_tensor():
    # Single qubit state |0><0|
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']

    # Two qubit state |00><00|
    rho_00=cb1.generate_entanglement_multi_qubit(rho_0,0,[0])

    # Error probabilities
    parameters= [0.1, 0.2, 0.3]

    # Apply error gate tensor
    result = cb1.error_gate_tensor(rho_00, parameters, 0, [0,1])

    # Manually compute expected result
    rho_0_q1 = cb1.error_gate(rho_0, parameters)
    rho_0_q2= rho_0
    expected = qt.tensor(rho_0_q1,rho_0_q2) 

    np.testing.assert_allclose(result.full(), expected.full(), rtol=1e-6, atol=1e-8)

def test_error_gate_tensor_single():
    # Single qubit state |0><0|
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']

    # Error probabilities
    parameters= [0.1, 0.2, 0.3]

    # Apply error gate tensor
    result = cb1.error_gate_tensor(rho_0, parameters, 0, [0])

    # Manually compute expected result
    expected = cb1.error_gate(rho_0, parameters)

    np.testing.assert_allclose(result.full(), expected.full(), rtol=1e-6, atol=1e-8)

def test_error_gate_tensor_entangled():
    parameters= [0.1, 0, 0]

    # Start with entangled state rho 0
    rho_0 = cb1.get_Z_basis_states()['RHO_ZERO']
    # Send it through a channel
    rho_0_prime=cb1.error_gate_tensor(rho_0, parameters,0,[0])
    rho_entangled= cb1.generate_entanglement_multi_qubit(rho_0_prime,0,[0])

    # We should now have the |r'r'><r'r'| + |rr><rr| mixed state.

    # For the given parameters we can solve for the system by hand:

    lhs = 0.1 * qt.tensor(qt.basis(2,1),qt.basis(2,1)).proj()
    rhs = 0.9 * qt.tensor(qt.basis(2,0),qt.basis(2,0)).proj()

    expected_state = lhs + rhs

    print(rho_entangled)

    print(expected_state)

    np.testing.assert_allclose(rho_entangled.full(), expected_state.full(), rtol=1e-6, atol=1e-8)


# ===========================================================================
# TEST MULTI QUBIT ENTANGLEMENT GATE WITH PERTURBATION
# ===========================================================================

def test_generate_entanglement_multi_qubit_dict_single_qubit():

    rho_dict= cb1.get_perturbed_0_state()
    entangled_rho_dict= cb1.generate_entanglement_multi_qubit_dict(rho_dict, 0, [0])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(entangled_rho_dict.keys()) == expected_keys

    
    expected_bell_state = Qobj([[1, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
    np.testing.assert_allclose(entangled_rho_dict['base'].full(), expected_bell_state.full(), rtol=1e-6, atol=1e-8)

    
def test_generate_entanglement_multi_qubit_dict_multi_qubit():
    rho_dict = cb1.get_perturbed_0_state()
    rho_dict_sup = cb1.apply_gate_rho_dict(rho_dict, cb1.get_all_gates()['H'])
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict_sup, 0, [0])

    three_qubit_entangled = cb1.generate_entanglement_multi_qubit_dict(entangled_rho_dict,0,[0,1])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(three_qubit_entangled.keys()) == expected_keys

    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)

    expected_state = (qt.tensor(ket0, ket0, ket0) + qt.tensor(ket1, ket1, ket1)).proj() / 2

    assert three_qubit_entangled['base'] == expected_state

    assert three_qubit_entangled['pX_plus'] == expected_state

    assert three_qubit_entangled['pY_minus'] == expected_state

    assert three_qubit_entangled['pZ_minus'] == expected_state

def test_generate_entanglement_multi_qubit_dict_complex():
    # Generate the state
    rho_dict = cb1.get_perturbed_0_state() # Initialize qubit

    rho_dict_prime = cb1.error_gate_perturbed(rho_dict, [0.1,0,0], h=1e-6) # Send it through a channel
    
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict_prime, 10, [10]) # Entangle it
    three_qubit_entangled = cb1.generate_entanglement_multi_qubit_dict(entangled_rho_dict,15,[10,15]) # Entangle it again

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(three_qubit_entangled.keys()) == expected_keys

    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)

    # The expected state should be |rrr><rrr| + |r'r'r'><r'r'r'|

    r_state= (qt.tensor(ket0, ket0, ket0))

    r_prime_state= (qt.tensor(ket1, ket1, ket1))


    expected_state_base = 0.1 * r_prime_state.proj() + 0.9 * r_state.proj()
    expected_state_plus = (0.1 + 1e-6) * r_prime_state.proj() + (0.9 - 1e-6) * r_state.proj()
    expected_state_minus = (0.1 - 1e-6) * r_prime_state.proj() + (0.9 + 1e-6) * r_state.proj()

    assert three_qubit_entangled['base'] == expected_state_base

    assert three_qubit_entangled['pX_plus'] == expected_state_plus
    assert three_qubit_entangled['pX_minus'] == expected_state_minus

# ===========================================================================
# TEST APPLY GATE MULTI QUBIT DICT FUNCTION
# ===========================================================================

def test_multi_qubit_apply_gate_dict():
    rho_dict = cb1.get_perturbed_0_state()
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict, 0, [0])

    H_gate= cb1.get_all_gates()['H']

    new_rho_dict= cb1.multi_qubit_apply_gate_dict(entangled_rho_dict, H_gate, 1, [0,1])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys

    ket_0 = qt.basis(2,0)
    ket_1 = qt.basis(2,1)

    expected_state = (qt.tensor(ket_0,ket_0) + qt.tensor(ket_0,ket_1)).proj() /2

    assert new_rho_dict['base'] == expected_state

def test_multi_qubit_apply_gate_dict_single_qubit():
    rho_dict = cb1.get_perturbed_0_state()

    X_gate= cb1.get_all_gates()['X']

    new_rho_dict= cb1.multi_qubit_apply_gate_dict(rho_dict, X_gate, 0, [0])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys

    assert new_rho_dict['base'] == cb1.get_Z_basis_states()['RHO_ONE']

def test_multi_qubit_apply_gate_dict_entangled():
    rho_dict = cb1.get_perturbed_0_state()
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict, 0, [0])

    H_gate= cb1.get_all_gates()['H']

    new_rho_dict= cb1.multi_qubit_apply_gate_dict(entangled_rho_dict, H_gate, 2, [0,2])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys

    ket_0 = qt.basis(2,0)
    ket_1 = qt.basis(2,1)

    expected_state = (qt.tensor(ket_0,ket_0) + qt.tensor(ket_0,ket_1)).proj() /2

    assert new_rho_dict['base'] == expected_state

def test_multi_qubit_apply_gate_dict_complex():
    # Generate the state
    rho_dict = cb1.get_perturbed_0_state()
    rho_dict_prime = cb1.error_gate_perturbed(rho_dict, [0.9,0,0], h=1e-6) 
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict_prime, 10, [10]) 
    three_qubit_entangled = cb1.generate_entanglement_multi_qubit_dict(entangled_rho_dict,15,[10,15]) 

    X_gate= cb1.get_all_gates()['X']

    new_rho_dict= cb1.multi_qubit_apply_gate_dict(three_qubit_entangled, X_gate, 15, [10,15,20])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys

    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)

    expected_base_state = 0.9 * qt.tensor(ket1, ket0, ket1).proj() + 0.1 * qt.tensor(ket0, ket1, ket0).proj()
    expected_plus_state =  (0.9 + 1e-6) * qt.tensor(ket1, ket0, ket1).proj() + (0.1 - 1e-6) * qt.tensor(ket0, ket1, ket0).proj()
    expected_minus_state =  (0.9 - 1e-6) * qt.tensor(ket1, ket0, ket1).proj() + (0.1 + 1e-6) * qt.tensor(ket0, ket1, ket0).proj()

    np.testing.assert_allclose(new_rho_dict['base'].full(), expected_base_state.full(),rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(new_rho_dict['pX_plus'].full(), expected_plus_state.full(),rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(new_rho_dict['pX_minus'].full(), expected_minus_state.full(),rtol=1e-6, atol=1e-8)

    
# ===========================================================================
# TEST MULTI QUBIT ERROR GATE MULTI QUBIT DICT FUNCTION
# ===========================================================================
    
def test_multi_qubit_error_gate_dict():
    rho_dict = cb1.get_perturbed_0_state()
    entangled_rho_dict = cb1.generate_entanglement_multi_qubit_dict(rho_dict, 0, [0])

    parameters= [0.1, 0.2, 0.3]

    new_rho_dict= cb1.multi_qubit_error_gate(entangled_rho_dict, parameters, 1, [0,1])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys
    
    
    density_matrix = np.array([
        [0.7+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.3+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j]
    ])

    density_matrix_X_minus = np.array([
        [0.700001+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.299999+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j],
        [0.00+0.j, 0.00+0.j, 0.00+0.j, 0.00+0.j]
    ])


    
    np.testing.assert_allclose(new_rho_dict['base'].full(), density_matrix, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(new_rho_dict['pX_minus'].full(), density_matrix_X_minus, rtol=1e-6, atol=1e-8)

def test_multi_qubit_error_gate_dict_single_qubit():
    rho_dict = cb1.get_perturbed_0_state()

    parameters= [0.1, 0.2, 0.3]

    new_rho_dict= cb1.multi_qubit_error_gate(rho_dict, parameters, 0, [0])

    expected_keys = {'pX_plus', 'pX_minus', 'pY_plus', 'pY_minus', 'pZ_plus', 'pZ_minus','base'}
    assert set(new_rho_dict.keys()) == expected_keys

    expected_state = cb1.error_gate(cb1.get_Z_basis_states()['RHO_ZERO'], parameters)

    np.testing.assert_allclose(new_rho_dict['base'].full(), expected_state.full(), rtol=1e-6, atol=1e-8)

# ===========================================================================
# TEST ACTION ENCODING FUNCTION
# ===========================================================================

def test_actions_to_gates_Initialization():
    prior_action_dict= {} # Empty prior actions
    action_vector = [1,2,3,3,2]
    parameters= [0.1,0.2,0.3]

    result = cb1.actions_to_gates(prior_action_dict, action_vector,parameters,DEBUG=True)

    print(result.keys())
    assert result.keys() == {0} # Just has one qubit
    assert result[0][0]['base'] == cb1.get_Z_basis_states()['RHO_ZERO'] # Initialized to |0><0|
    assert result[0][2]==2 # Location should be 2

    prior_action_dict2 ={}
    action_vector2 = [0,1,2,3,3,3]

    result2 = cb1.actions_to_gates(prior_action_dict2, action_vector2,parameters,DEBUG=True)
    assert result2.keys() == {0} # Just has one qubit
    assert result2[0][0]['base'] == cb1.get_Z_basis_states()['RHO_ZERO']
    assert result2[0][2]==3 # Location should be 3

def test_actions_to_gates_action0():
    
    prior_action_dict = cb1.get_test_action_dict()
    # Quantum Circuit description:
    # Qubit 0: |0> state at node 1
    # Qubit 1: |+> state at node 2
    # Qubit 2 and 3: Entangled pair at node 3

    action_vector = [0, # Apply Gate
                     1, # X gate
                     0, # Qubit 0
                     3, # NA
                     2] # NA
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, [0.1,0.2,0.3],DEBUG=True)

    assert new_action_dict.keys() == {0,1,2,3} # Four qubits
    # Qubit 0 should have X gate applied at node 1
    expected_rho_1 = cb1.get_perturbed_0_state()
    expected_rho_1 = cb1.multi_qubit_apply_gate_dict(expected_rho_1, cb1.get_all_gates()['X'], 0, [0])

    assert new_action_dict[0][0] == expected_rho_1 

def test_actions_to_gates_action0_entangled():
    prior_action_dict = cb1.get_test_action_dict_entangled()
    # Quantum Circuit description:
    # Qubit 0: |0> state at node 2
    # Qubit 1: |0> state at node 0
    # Both qubits are entangled

    action_vector = [0, # apply gate
                     1, # X gate
                     1, # Qubit 1
                     2, # NA
                     2] # NA
    
    params= [[0.1,0.2,0.3],[0.2,0.15,0.2],[0.05,0.1,0.2]] # Different error parameters for different channels

    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1} # Two qubits
    # Qubit 1 should be entangled with qubit 2 at node 2

    q0_loc= new_action_dict[0][2]
    q0_ent_vec= new_action_dict[0][1]
    q0_state= new_action_dict[0][0]

    q1_loc= new_action_dict[1][2]
    q1_ent_vec= new_action_dict[1][1]
    q1_state= new_action_dict[1][0]

    assert q0_loc ==2
    assert q1_loc ==0

    assert q0_ent_vec == [0,1]
    assert q1_ent_vec == [0,1]

    expected_rho_1 = cb1.get_perturbed_0_state()
    expected_rho_1 = cb1.generate_entanglement_multi_qubit_dict(expected_rho_1, 0, [0])
    expected_rho_1 = cb1.multi_qubit_apply_gate_dict(expected_rho_1, cb1.get_all_gates()['X'], 1, [0,1])

    assert q0_state == expected_rho_1
    assert q1_state == expected_rho_1

def test_actions_to_gates_action1_case1():
    # The qubit wants to move to an edge node that it is already located at.
    prior_action_dict = cb1.get_test_action_dict()

    action_vector = [1, # Move
                     0, # NA
                     0, # Qubit 0
                     1, # Move to node 1
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= prior_action_dict[0][0] # First element is rho dict
    q0_location= prior_action_dict[0][2] # Third element is location
    q0_entanglement_vector= prior_action_dict[0][1] # second element is entanglement vector
    assert new_action_dict.keys() == {0,1,2,3} # Four qubits
    assert q0_location ==1
    assert q0_entanglement_vector == [0]
    # Qubit 0 should remain at node 1 with no changes
    expected_state = cb1.multi_qubit_error_gate(q0_state, [0.1,0.2,0.3], 0, [0])
    expected_state = cb1.multi_qubit_error_gate(expected_state, [0.1,0.2,0.3], 0, [0])

    print("Booyah")
    
    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        print(new_action_dict[0][0][key])

def test_actions_to_gates_action1_case1_entangled():
    # The qubit wants to move to an edge node that it is already located at.
    prior_action_dict = cb1.get_test_action_dict_entangled()

    action_vector = [1, # Move
                     0, # NA
                     0, # Qubit 0
                     2, # Move to node 2
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)
    q0_state_prior= prior_action_dict[0][0] # First element is rho dict

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element is entanglement vector

    q1_state= new_action_dict[1][0] # First element is rho dict
    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    assert new_action_dict.keys() == {0,1} # Two qubits
    assert q0_location ==2
    assert q0_entanglement_vector == [0,1]

    assert q1_location ==0
    assert q1_entanglement_vector == [0,1]

    # Qubit 0 should remain at node 2 with channel 2 applied twice
    expected_state = cb1.multi_qubit_error_gate(q0_state_prior, [0.2,0.15,0.2], 0, [0,1])
    expected_state = cb1.multi_qubit_error_gate(expected_state, [0.2,0.15,0.2], 0, [0,1])

    print("Booyah")
    
    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_action_dict[1][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case2():
    prior_action_dict = cb1.get_test_action_dict()
    # Add another qubit at node 0 for testing
    q4= cb1.get_perturbed_0_state()
    prior_action_dict[4] = [q4, [4], 0] # New qubit at node 0

    action_vector = [1, # Move
                     0, # NA
                     4, # Qubit 4
                     1, # Move to node 1
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q4_state= new_action_dict[4][0] # First element is rho dict
    q4_location= new_action_dict[4][2] # Third element is location
    q4_entanglement_vector= new_action_dict[4][1] # second element

    assert new_action_dict.keys() == {0,1,2,3,4} # Five qubits
    assert q4_location ==1 # Moved to node 1
    assert q4_entanglement_vector == [4] # Entanglement vector unchanged

    # Qubit 4 should have moved to node 1 with error channels applied
    expected_state = cb1.multi_qubit_error_gate(q4, [0.1,0.2,0.3], 4, [4])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[4][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
    
def test_actions_to_gates_action1_case2_entangled():
    # The qubit is moved from the center to the edge
    prior_action_dict = cb1.get_test_action_dict_entangled()

    action_vector = [1, # Move
                     0, # NA
                     1, # Qubit 1
                     2, # Move to node 2
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q1_state= new_action_dict[1][0] # First element is rho dict
    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    assert new_action_dict.keys() == {0,1} # two qubits
    assert q1_location ==2 # Should now be at node 2
    assert q1_entanglement_vector == [0,1] # Entanglement vector unchanged

    q1_prior_state= prior_action_dict[1][0]
    # Qubit 2 should have moved to node 2 with error channels applied
    expected_state = cb1.multi_qubit_error_gate(q1_prior_state, [0.2,0.15,0.2], 1, [0,1])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_action_dict[1][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case3():
    """" Test moving a qubit from an edge node to a central node"""

    # Initialize prior action dict with one qubit at node 1
    prior_action_dict = cb1.get_test_action_dict()

    # Here we will move qubit 0 from node 1 to central node (node 0)

    action_vector = [1, # Move
                    0, # NA
                    0, # Qubit 0
                    0, # Move to node 0 (central)
                    2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element

    assert new_action_dict.keys() == {0,1,2,3} # Four qubits
    assert q0_location ==0 # Moved to central node
    assert q0_entanglement_vector == [0] # Entanglement vector unchanged

    # Qubit 0 should have moved to central node with error channels applied
    expected_state = cb1.multi_qubit_error_gate(prior_action_dict[0][0], [0.1,0.2,0.3], 0, [0])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case3_entangled():
    """" Test moving a qubit from an edge node to a central node"""

    # Initialize prior action dict with one qubit at node 2
    prior_action_dict = cb1.get_test_action_dict_entangled()

    # Here we will move qubit 0 from node 2 to central node (node 0)

    action_vector = [1, # Move
                    0, # NA
                    0, # Qubit 0
                    0, # Move to node 0 (central)
                    2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element

    q1_state= new_action_dict[1][0] # First element is rho dict
    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    assert new_action_dict.keys() == {0,1} # Two qubits
    assert q0_location ==0 # Moved to central node
    assert q0_entanglement_vector == [0,1] # Entanglement vector unchanged

    assert q1_location ==0 # Should still be at central node
    assert q1_entanglement_vector == [0,1] # Entanglement vector


    # Qubit 0 should have moved to central node with error channels applied
    expected_state = cb1.multi_qubit_error_gate(prior_action_dict[0][0], [0.2,0.15,0.2], 0, [0,1])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_action_dict[1][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case4():
    """" Test moving a qubit from a central node back to the central node
        Currently how this is handled is that the qubit experiences no change.
    """

    # Initialize prior action dict with one qubit at central node
    prior_action_dict = cb1.get_test_action_dict()
    prior_action_dict[4] = [cb1.get_perturbed_0_state(), [4], 0] # New qubit at central node

    # Here we will move qubit 5 from node 0 to edge node 2

    action_vector = [1, # Move
                    0, # NA
                    4, # Qubit 4
                    0, # Move to node 0 center
                    2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    new_state= new_action_dict[4][0] # First element is rho dict
    new_location= new_action_dict[4][2] # Third element is location
    new_entanglement_vector= new_action_dict[4][1] # second element

    expected_state = cb1.get_perturbed_0_state()
    expected_location = 0
    expected_entanglement_vector = [4]

    assert new_action_dict.keys() == {0,1,2,3,4} # Five qubits
    assert new_location == expected_location # Location unchanged
    assert new_entanglement_vector == expected_entanglement_vector # Entanglement vector unchanged

    for key in expected_state.keys():
        np.testing.assert_allclose(new_state[key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
    
    assert prior_action_dict == new_action_dict

def test_actions_to_gates_action1_case4_entangled():
    """ Test moving a qubit from a central node back to the central node
        Currently how this is handled is that the qubit experiences no change.
    """

    # Initialize prior action dict with one qubit at central node
    prior_action_dict = cb1.get_test_action_dict_entangled()

    # Here we will move qubit 0 from node 0 to node 0

    action_vector = [1, # Move
                    0, # NA
                    1, # Qubit 0
                    0, # Move to node 0 center
                    2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    new_state_q0= new_action_dict[0][0] # First element is rho dict
    new_location_q0= new_action_dict[0][2] # Third element is location
    new_entanglement_vector_q0= new_action_dict[0][1] # second element

    new_state_q1= new_action_dict[1][0] # First element is rho dict
    new_location_q1= new_action_dict[1][2] # Third element is location
    new_entanglement_vector_q1= new_action_dict[1][1] # second element

    expected_state = prior_action_dict[0][0]
    expected_location_q0 = 2
    expected_location_q1 = 0
    expected_entanglement_vector = [0,1]

    assert new_action_dict.keys() == {0,1} # Two qubits
    assert new_location_q0 == expected_location_q0 # Location unchanged
    assert new_location_q1 == expected_location_q1 # Location unchanged
    assert new_entanglement_vector_q0 == expected_entanglement_vector # Entanglement vector unchanged
    assert new_entanglement_vector_q1 == expected_entanglement_vector # Entanglement vector unchanged

    for key in expected_state.keys():
        np.testing.assert_allclose(new_state_q0[key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_state_q1[key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case5():
    """" Test moving a qubit from an edge node to another edge node

        This is handled by first moving to the central node with one error channel,
        then moving to the target edge node with another error channel.
    """

    # Initialize prior action dict with one qubit at node 1
    prior_action_dict = cb1.get_test_action_dict()

    # Here we will move qubit 0 from node 1 to edge node 2
    action_vector = [1, # Move
                    0, # NA
                    0, # Qubit 0
                    2, # Move to node 2
                    2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element

    assert new_action_dict.keys() == {0,1,2,3} # Four qubits
    assert q0_location ==2 # Moved to node 2
    assert q0_entanglement_vector == [0] # Entanglement vector unchanged

    # Qubit 0 should have moved to node 2 with error channels applied
    expected_state = cb1.multi_qubit_error_gate(prior_action_dict[0][0], [0.1,0.2,0.3], 0, [0])
    expected_state = cb1.multi_qubit_error_gate(expected_state, [0.2,0.15,0.2], 0, [0])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case5_entangled():
    """ Test moving a qubit from an edge node to another edge node

        This is handled by first moving to the central node with one error channel,
        then moving to the target edge node with another error channel.
    """

    # Initialize prior action dict with one qubit at node 2
    prior_action_dict = cb1.get_test_action_dict_entangled()

    # Here we will move qubit 0 from node 2 to edge node 1
    action_vector = [1, # Move
                    0, # NA
                    0, # Qubit 0
                    1, # Move to node 1
                    2] # NA
    # Qubit will go from location 2 -> 0 -> 1
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element

    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    assert new_action_dict.keys() == {0,1} # Two qubits
    assert q0_location ==1 # Moved to node 1
    assert q0_entanglement_vector == [0,1] # Entanglement vector unchanged

    assert q1_location ==0 # Should still be at central node
    assert q1_entanglement_vector == [0,1] # Entanglement vector unchanged

    # Qubit 0 should have moved to node 1 with error channels applied
    expected_state = cb1.multi_qubit_error_gate(prior_action_dict[0][0], [0.2,0.15,0.2], 0, [0,1])
    expected_state = cb1.multi_qubit_error_gate(expected_state, [0.1,0.2,0.3], 0, [0,1])

    for key in expected_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_action_dict[1][0][key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action2():
    """ Test adding a new qubit to the circuit"""
    prior_action_dict = cb1.get_test_action_dict()
    # Qubit 0: |0> state at node 1
    # Qubit 1: |+> state at node 2
    # Qubit 2 and 3: Entangled pair at node 3

    action_vector = [2, # Add
                     0, # NA
                     1, # NA
                     3, # NA
                     2] # Source Location at node 2
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1,2,3,4} # Another qubit is added

    assert new_action_dict[0] == prior_action_dict[0] # Qubit 0 unchanged
    assert new_action_dict[1] == prior_action_dict[1] # Qubit 1 unchanged
    assert new_action_dict[2] == prior_action_dict[2] # Qubit 2 unchanged
    assert new_action_dict[3] == prior_action_dict[3] # Qubit 3 unchanged

    q4_loc = new_action_dict[4][2]
    q4_ent_vec = new_action_dict[4][1]
    q4_state = new_action_dict[4][0]

    assert q4_loc ==2
    assert q4_ent_vec == [4] # New qubit entanglement vector
    expected_state = cb1.get_perturbed_0_state() # New qubit state

    for key in expected_state.keys():
        np.testing.assert_allclose(q4_state[key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)
    
def test_actions_to_gates_action3_non_entangled():
    """ Test entangling a non-entangled qubit with another qubit"""

    prior_action_dict = cb1.get_test_action_dict()
    # Qubit 0: |0> state at node 1
    # Qubit 1: |+> state at node 2
    # Qubit 2 and 3: Entangled pair at node 3

    action_vector = [3, # Entangle
                     0, # NA
                     0, # Qubit 0
                     3, # NA
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1,2,3,4} # Five qubits since we added one more

    q0_loc = new_action_dict[0][2]
    q0_ent_vec = new_action_dict[0][1]

    q4_loc = new_action_dict[4][2]
    q4_ent_vec = new_action_dict[4][1]

    assert q0_loc ==1
    assert q0_ent_vec == [0,4] # Now entangled with new qubit 4

    assert q4_loc ==1
    assert q4_ent_vec == [0,4] # Entangled with qubit 0

    entangled_state = cb1.get_perturbed_0_state()
    entangled_state = cb1.generate_entanglement_multi_qubit_dict(entangled_state,0,[0])

    for key in entangled_state.keys():
        np.testing.assert_allclose(new_action_dict[0][0][key].full(), entangled_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(new_action_dict[4][0][key].full(), entangled_state[key].full(), rtol=1e-6, atol=1e-8)
    
def test_actions_to_gates_action3_entangled():
    """ Test entangling an already entangled qubit with another qubit"""

    prior_action_dict = cb1.get_test_action_dict()
    # Qubit 0: |0> state at node 1
    # Qubit 1: |+> state at node 2
    # Qubit 2 and 3: Entangled pair
    # Qubit 2 at node 3 and Qubit 3 at node 0

    action_vector = [3, # Entangle
                     0, # NA
                     2, # Qubit 2 (entangled)
                     3, # NA
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1,2,3,4} # Five qubits since we added one more

    q2_loc = new_action_dict[2][2]
    q2_ent_vec = new_action_dict[2][1]
    q2_state = new_action_dict[2][0]

    q4_loc = new_action_dict[4][2]
    q4_ent_vec = new_action_dict[4][1]
    q4_state = new_action_dict[4][0]

    q3_ent_vec = new_action_dict[3][1]
    q3_loc = new_action_dict[3][2]
    q3_state = new_action_dict[3][0]

    print(q3_loc)

    assert q2_loc ==3
    assert q2_ent_vec == [2,3,4] # Now entangled with new qubit 4

    assert q4_loc ==3
    assert q4_ent_vec == [2,3,4] # Entangled with qubit 2

    assert q3_ent_vec == [2,3,4] # Qubit 3 entanglement vector also updated
    assert q3_loc == 0

    for key in q2_state.keys():
        np.testing.assert_allclose(q2_state[key].full(), q4_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(q2_state[key].full(), q3_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action4():
    """ Test measuring a qubit this is handled by sending
        all qubits in the center channel to the end node.
    """

    prior_action_dict = cb1.get_test_action_dict()
    # Qubit 0: |0> state at node 1
    # Qubit 1: |+> state at node 2
    # Qubit 2 and 3: Entangled pair at node 3

    action_vector = [4, # Measure ALL
                     0, # NA
                     2, # NA
                     3, # NA
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1,2,3} # Four qubits
    # All qubits should now be at their respective end nodes
    assert new_action_dict[0][2] ==1
    assert new_action_dict[1][2] ==2
    assert new_action_dict[2][2] ==3
    assert new_action_dict[3][2] ==1 # Should have moved from center to end node 1


    # Testing the state of qubit 3 and qubit 2
    q2_state= new_action_dict[2][0]
    q3_state= new_action_dict[3][0]


    for key in q2_state.keys():
        np.testing.assert_allclose(q2_state[key].full(), q3_state[key].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action4_entangled():
    """ Test measuring a qubit this is handled by sending
        all qubits in the center channel to the end node.
    """

    prior_action_dict = cb1.get_test_action_dict_entangled()
    # Qubit 0: |0> state at node 2
    # Qubit 1: |0> state at node 0
    # Both qubits are entangled

    action_vector = [4, # Measure ALL
                     0, # NA
                     0, # NA
                     3, # NA
                     2] # NA
    
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    assert new_action_dict.keys() == {0,1} # Two qubits
    # All qubits should now be at their respective end nodes
    assert new_action_dict[0][2] ==2
    assert new_action_dict[1][2] ==1


    # Testing the state of qubit 0 and qubit 1
    q0_state= new_action_dict[0][0]
    q1_state= new_action_dict[1][0]

    expected_state=prior_action_dict[1][0]
    expected_state= cb1.multi_qubit_error_gate(expected_state, [0.1,0.2,0.3], 1, [0,1])
    # The qubit at the center node should have moved to node 1.
    for key in q0_state.keys():
        np.testing.assert_allclose(q0_state[key].full(), q1_state[key].full(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(q0_state[key].full(), expected_state[key].full(), rtol=1e-6, atol=1e-8)

# ===========================================================
# CALCULATE QFIM TESTS
# ===========================================================

def test_calculate_qfim_single_qubit():
    base_rho = cb1.get_Z_basis_states()["RHO_ZERO"]
    p_qobj=cb1.P_QOBJ(base_rho,1,[0],0)


    params=[[0.1,0.2,0.3],
            [0.09,0.5,0.3],
            [0.01,0.2,0.09]]  # Example parameters

    p_qobj.configure_parameters(parameters = params,delta = 1e-6)
    p_qobj.apply_error(1)
    p_qobj.apply_error(2)
    p_qobj.apply_error(3)

    # p_qobj.apply_gate(cb1.get_all_gates()['H'])
    # p_qobj.apply_error(1)
    # p_qobj.apply_error(2)
    # p_qobj.apply_error(3)

    # p_qobj.apply_gate(cb1.get_all_gates()['H'])
    # p_qobj.apply_gate(cb1.get_all_gates()['S'])
    # p_qobj.apply_error(1)
    # p_qobj.apply_error(2)
    # p_qobj.apply_error(3)

    rho_dict= p_qobj.rho_dicts

    delta = p_qobj.delta
    rho_dict_list = [rho_dict['channel_1'], rho_dict['channel_2'], rho_dict['channel_3']]

    qfim = cb1.calculate_QFIM_direct_singular(rho_dict_list,h=delta,debug=True)
    print("QFIM Single Qubit Test:")
    
    for row in qfim:
        print(row)

    # Test basic properties of QFIM
    assert qfim.shape == (9,9) 

    assert np.allclose(qfim,qfim.T, rtol=1e-6, atol=1e-8)  # Symmetric
    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvals(qfim)
    assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    for i in range(9):
        assert qfim[i,i] >=0  # Diagonal elements non-negative

def test_calculate_qfim_single_qubit_accuracy():
    base_rho = cb1.get_Z_basis_states()["RHO_ZERO"]
    p_qobj=cb1.P_QOBJ(base_rho,1,[0],0)


    params=[[0.1,0,0],
            [0.2,0,0],
            [0.3,0,0]]  # Example parameters

    p_qobj.configure_parameters(parameters = params,delta = 1e-6)
    p_qobj.apply_error(1) 
    p_qobj.apply_error(2)
    p_qobj.apply_error(3)

    rho_dict= p_qobj.rho_dicts

    delta = p_qobj.delta
    rho_dict_list = [rho_dict['channel_1'], rho_dict['channel_2'], rho_dict['channel_3']]

    # Calculate using cb1 function
    qfim_cb1 = cb1.calculate_QFIM_direct_singular(rho_dict_list,h=delta,debug=False)


    # calculate using qm2 function to test accuracy
    rho_dict= p_qobj.rho_dicts
    rho_list = [[rho_dict['channel_1']['base'].full()],
                [rho_dict['channel_1']['pX_plus'].full(),rho_dict['channel_1']['pX_minus'].full()],
                [rho_dict['channel_2']['pX_plus'].full(),rho_dict['channel_2']['pX_minus'].full()],
                [rho_dict['channel_3']['pX_plus'].full(),rho_dict['channel_3']['pX_minus'].full()]
                ]
    

    qfim_qm2 = qm2.calculate_QFIM_direct_singular(rho_list, h=delta)
    qfim_standard = np.array(qfim_qm2)
    print(qfim_standard)

    # Get 3 x 3 version of the qfim from cb1
    qfim_cb1_reduced = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            qfim_cb1_reduced[i,j] = qfim_cb1[i*3,j*3]

    # Compare the two QFIMs
    np.testing.assert_allclose(qfim_cb1_reduced, qfim_standard, rtol=1e-5, atol=1e-7)
 
def test_calculate_qfim_single_qubit_complex():
    base_rho = cb1.get_Z_basis_states()["RHO_ZERO"]
    p_qobj=cb1.P_QOBJ(base_rho,1,[0],0)

    params=[[0.1,0.2,0.3],
            [0.05,0.02,0.03],
            [0.08,0.4,0.09]]  # Example parameters
    
    p_qobj.configure_parameters(parameters = params,delta = 1e-6)

    p_qobj.apply_error(1)
    p_qobj.apply_error(2)
    p_qobj.apply_error(3)


    rho_dict= p_qobj.rho_dicts
    rho_dict_list = [rho_dict['channel_1'], rho_dict['channel_2'], rho_dict['channel_3']]
    
    delta = p_qobj.delta

    qfim_cb1= cb1.calculate_QFIM_direct_singular(rho_dict_list,h=delta)

    # Test basic properties of QFIM
    assert qfim_cb1.shape == (9,9)
    assert np.allclose(qfim_cb1,qfim_cb1.T, rtol=1e-6, atol=1e-8)  # Symmetric
    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvals(qfim_cb1)
    assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    for i in range(9):
        assert qfim_cb1[i,i] >=0  # Diagonal elements non-negative

    
    rho2=cb1.get_Z_basis_states()["RHO_ZERO"]
    p_qobj2=cb1.P_QOBJ(rho2,1,[0],0)

    p_qobj2.apply_error(1)
    p_qobj2.apply_error(2)
    p_qobj2.apply_error(3)

    h=cb1.get_all_gates()['H']
    p_qobj2.apply_gate(h)

    rho_dict2= p_qobj2.rho_dicts
    rho_dict_list2 = [rho_dict2['channel_1'], rho_dict2['channel_2'], rho_dict2['channel_3']]

    qfim_cb1_2= cb1.calculate_QFIM_direct_singular(rho_dict_list2,h=delta)

    np.testing.assert_allclose(qfim_cb1, qfim_cb1_2, rtol=1e-6, atol=1e-8)

def test_calculate_qfim_single_qubit_multi_qubit():
    params=[[0.1,0.2,0.3],
            [0.05,0.02,0.03],
            [0.08,0.4,0.09]]  # Example parameters
    
    dc= cb1.DistributionCircuit(params,1e-6)

    dc.add_qubit(1)# Adds a qubit to node 1
    dc.apply_error(0,1)
    dc.generate_entanglement(0) # Entangles qubit 0 with another qubit
    dc.apply_error(0,2)
    dc.apply_error(1,1)
    
    rho_dict=dc.get_rho_dicts(0) # Get rho dict for qubit 0

    rho_dict_list = [rho_dict['channel_1'], rho_dict['channel_2'], rho_dict['channel_3']]
    
    delta = 1e-6

    qfim_cb1= cb1.calculate_QFIM_direct_singular(rho_dict_list,h=delta)

    # Test basic properties of QFIM
    assert qfim_cb1.shape == (9,9)
    assert np.allclose(qfim_cb1,qfim_cb1.T, rtol=1e-6, atol=1e-8)  # Symmetric
    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvals(qfim_cb1)
    assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors
    for i in range(9):
        assert qfim_cb1[i,i] >=0  # Diagonal elements non-negative


    rho_dict_2 = dc.get_rho_dicts(1) # Get rho dict for qubit 1
    rho_dict_list_2 = [rho_dict_2['channel_1'], rho_dict_2['channel_2'], rho_dict_2['channel_3']]
    qfim_cb1_2= cb1.calculate_QFIM_direct_singular(rho_dict_list_2,h=delta)

    np.testing.assert_allclose(qfim_cb1, qfim_cb1_2, rtol=1e-6, atol=1e-8)


    # Make another quantum system with added unitary operations

    dc2= cb1.DistributionCircuit(params,1e-6)
    dc2.add_qubit(1)# Adds a qubit to node 1
    dc2.apply_error(0,1)
    dc2.generate_entanglement(0) # Entangles qubit 0 with another qubit
    dc2.apply_error(0,2)
    dc2.apply_error(1,1)

    dc2.apply_gate_to_qubit(0,4)
    dc2.apply_gate_to_qubit(1,6)

    rho_dict_3 = dc2.get_rho_dicts(0) # Get rho dict for qubit 0
    rho_dict_list_3 = [rho_dict_3['channel_1'], rho_dict_3['channel_2'], rho_dict_3['channel_3']]
    qfim_cb1_3= cb1.calculate_QFIM_direct_singular(rho_dict_list_3,h =delta)  

    np.testing.assert_allclose(qfim_cb1, qfim_cb1_3, rtol=1e-6, atol=1e-8)



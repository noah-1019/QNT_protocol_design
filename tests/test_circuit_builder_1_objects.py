"""
A script containing unit tests for the circuit_builder_1 module. To conduct these
tests at any times enter pytest tests/test_circuit_builder_1_objects.py in the terminal.
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


# ===========================================================================
# TEST GET TEST CIRCUIT
# ===========================================================================

def test_get_test_circuit():
    """ Test the get_test_circuit function to ensure it returns a valid CircuitBuilder1 object. """
    circuit = cb1.get_test_circuit()
    assert isinstance(circuit, cb1.DistributionCircuit), "The returned object is not an instance of CircuitBuilder1."


# ===========================================================================
# TEST P_QOBJ METHODS
# ===========================================================================

def test_p_qobj_properties():
    """ Test the properties of the P_QOBJ class. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0], qubit_id=0)

    
    base_rho = p_qobj.channel_1['base']
    base_2_rho = p_qobj.channel_2['base']
    base_3_rho = p_qobj.channel_3['base']

    assert base_2_rho == rho, "Density matrix does not match the initialized value."
    assert base_3_rho == rho, "Density matrix does not match the initialized value"
    assert base_rho == rho, "Density matrix does not match the initialized value."

    assert p_qobj.location == 0, "Source location does not match."
    assert p_qobj.entanglement_vector == [0], "Entanglement vector does not match."
    assert p_qobj.qubit_id == 0, "Qubit ID does not match."

def test_p_qobj_setters():
    """ Test the setters of the P_QOBJ class. """
    rho= cb1.get_Z_basis_states()['RHO_ZERO']
    rho_1 = cb1.get_Z_basis_states()['RHO_ONE']

    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0], qubit_id=0)
    p_qobj_1 = cb1.P_QOBJ(rho_1, location=1, entanglement_vector=[1], qubit_id=1)

    p_qobj.location = 1
    p_qobj.entanglement_vector = [2]
    p_qobj.qubit_id = 2

    assert p_qobj.location == 1, "Source location setter did not work."
    assert p_qobj.entanglement_vector == [2], "Entanglement vector setter did not work."
    assert p_qobj.qubit_id == 2, "Qubit ID setter did not work."

    p_qobj.set_state(p_qobj_1)

    assert p_qobj.channel_1['base'] == rho_1, "Density matrix setter did not work."

def test_p_qobj_apply_gate_method():
    """ Test the apply_gate method of the P_QOBJ class. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0], qubit_id=0)

    # Apply a Hadamard gate
    h=cb1.get_all_gates()['H']
    p_qobj.apply_gate(h)

    expected_state=cb1.get_perturbed_0_state()
    expected_state = cb1.multi_qubit_apply_gate_dict(expected_state,h,qubit_id=0,entanglement_vector=[0])

    states=p_qobj.rho_dicts

    for i in states.keys():
        for k in states[i].keys():
            np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Gate application did not produce expected state.")

def test_p_qobj_apply_gate_method_entangled():
    """ Test the apply_gate method of the P_QOBJ class for entangled qubits. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    rho_plus = cb1.get_Z_basis_states()['RHO_ZERO']
    combined_rho = qt.tensor(rho, rho_plus)

    p_qobj = cb1.P_QOBJ(combined_rho, location=0, entanglement_vector=[0, 1], qubit_id=0)

    # Apply a CNOT gate with qubit 0 as control and qubit 1 as target
    h = cb1.get_all_gates()['H']

    p_qobj.apply_gate(h)


    rho_perturbed = cb1.get_perturbed_0_state()
    combined_rho_expected = cb1.generate_entanglement_multi_qubit_dict(rho_perturbed, 0, [0])
    expected_state = cb1.multi_qubit_apply_gate_dict(combined_rho_expected, h, qubit_id=0, entanglement_vector=[0, 1])

    states = p_qobj.rho_dicts

    for i in states.keys():
        for k in states[i].keys():
            np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Gate application on entangled qubits did not produce expected state.")
    
def test_p_qobj_generate_entanglement_method():
    """ Test the generate_entanglement method of the P_QOBJ class. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0], qubit_id=0)
    p_qobj.generate_entanglement()

    rho_perturbed = cb1.get_perturbed_0_state()
    expected_state = cb1.generate_entanglement_multi_qubit_dict(rho_perturbed, 0, [0])

    states = p_qobj.rho_dicts

    for i in states.keys():
        for k in states[i].keys():
            np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Entanglement generation did not produce expected state.")  

def test_p_qobj_generate_entanglement_method_entangled():
    """ Test the generate_entanglement method of the P_QOBJ class for already entangled qubits. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    rho_plus = cb1.get_Z_basis_states()['RHO_ZERO']
    combined_rho = qt.tensor(rho, rho_plus)

    p_qobj = cb1.P_QOBJ(combined_rho, location=0, entanglement_vector=[0, 1], qubit_id=0)
    p_qobj.generate_entanglement()

    rho_perturbed = cb1.get_perturbed_0_state()
    combined_rho_expected = cb1.generate_entanglement_multi_qubit_dict(rho_perturbed, 0, [0])
    expected_state = cb1.generate_entanglement_multi_qubit_dict(combined_rho_expected, 0, [0, 1])

    states = p_qobj.rho_dicts

    for i in states.keys():
        for k in states[i].keys():
            np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Entanglement generation on already entangled qubits did not produce expected state.")

def test_p_qobj_apply_error_method():
    """ Test the apply_gate method of the P_QOBJ class. """
    rho = cb1.get_Z_basis_states()['RHO_ZERO']
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0], qubit_id=0)

    parameters=[[0.1,0.2,0.3],[0.01,0.02,0.03],[0.01,0.02,0.03]]
    p_qobj.configure_parameters(parameters=parameters,delta=1e-6)

    p_qobj.apply_error(1)

    expected_state=cb1.get_perturbed_0_state()
    expected_state=cb1.multi_qubit_error_gate(expected_state,[0.1,0.2,0.3],0,[0])

    expected_state_non_perturbed=cb1.multi_qubit_error_gate_non_perturbed(cb1.get_perturbed_0_state(),[0.1,0.2,0.3],0,[0])
    states=p_qobj.rho_dicts
    for i in states.keys():
        for k in states[i].keys():
            
            if i== 'channel_1':
                np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Gate application did not produce expected state.")
            else:
                print(expected_state_non_perturbed[k].full())
                np.testing.assert_allclose(states[i][k].full(), expected_state_non_perturbed[k].full(), err_msg="Gate application did not produce expected state.")
               
def test_p_qobj_apply_error_method_entangled():
    rho1 = cb1.get_Z_basis_states()['RHO_ZERO']
    rho2= cb1.get_Z_basis_states()['RHO_ZERO']

    rho= qt.tensor(rho1,rho2)
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0,1], qubit_id=1)

    parameters=[[0.1,0.2,0.3],[0.01,0.02,0.03],[0.01,0.02,0.03]]
    p_qobj.configure_parameters(parameters=parameters,delta=1e-6)

    p_qobj.apply_error(1)

    expected_state=cb1.get_perturbed_0_state()
    expected_state=cb1.generate_entanglement_multi_qubit_dict(expected_state,0,[0])
    expected_state=cb1.multi_qubit_error_gate(expected_state,[0.1,0.2,0.3],1,[0,1])

    non_p_expected=cb1.get_perturbed_0_state()
    non_p_expected=cb1.generate_entanglement_multi_qubit_dict(non_p_expected,0,[0])
    non_p_expected=cb1.multi_qubit_error_gate_non_perturbed(non_p_expected,[0.1,0.2,0.3],1,[0,1])

    states=p_qobj.rho_dicts
    for i in states.keys():
        for k in states[i].keys():
            
            if i== 'channel_1':
                np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Gate application did not produce expected state.")
            else:
                np.testing.assert_allclose(states[i][k].full(), non_p_expected[k].full(), err_msg="Gate application did not produce expected state.")

def test_p_qobj_apply_error_method_entangledX():
    rho1 = cb1.get_Z_basis_states()['RHO_ZERO']
    rho2= cb1.get_Z_basis_states()['RHO_ZERO']

    rho= qt.tensor(rho1,rho2)
    p_qobj = cb1.P_QOBJ(rho, location=0, entanglement_vector=[0,1], qubit_id=1)
    x_gate=cb1.get_all_gates()['X']
    p_qobj.apply_gate(x_gate)

    parameters=[[0.1,0.2,0.3],[0.01,0.02,0.03],[0.01,0.02,0.03]]
    p_qobj.configure_parameters(parameters=parameters,delta=1e-6)

    p_qobj.apply_error(1)

    expected_state=cb1.get_perturbed_0_state()
    expected_state=cb1.generate_entanglement_multi_qubit_dict(expected_state,0,[0])
    expected_state=cb1.multi_qubit_apply_gate_dict(expected_state,x_gate,1,[0,1])
    expected_state=cb1.multi_qubit_error_gate(expected_state,[0.1,0.2,0.3],1,[0,1])

    non_p_expected=cb1.get_perturbed_0_state()
    non_p_expected=cb1.generate_entanglement_multi_qubit_dict(non_p_expected,0,[0])
    non_p_expected=cb1.multi_qubit_apply_gate_dict(non_p_expected,x_gate,1,[0,1])
    non_p_expected=cb1.multi_qubit_error_gate_non_perturbed(non_p_expected,[0.1,0.2,0.3],1,[0,1])

    states=p_qobj.rho_dicts
    for i in states.keys():
        for k in states[i].keys():
            
            if i== 'channel_1':
                np.testing.assert_allclose(states[i][k].full(), expected_state[k].full(), err_msg="Gate application did not produce expected state.")
            else:
                np.testing.assert_allclose(states[i][k].full(), non_p_expected[k].full(), err_msg="Gate application did not produce expected state.")

# ===========================================================================
# TEST Distribution Circuit METHODS
# ===========================================================================

def test_distribution_circuit_getters_setters():
    dc = cb1.DistributionCircuit(channel_parameters=[[0.01, 0.02, 0.03],
                                                          [0.04, 0.05, 0.06],
                                                          [0.07, 0.08, 0.09]],
                                       delta=1e-6)
    
    params=[[0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09]]
    

    # Add qubit 0
    dc.add_qubit(1) # Node 1 # by default in |0> state
    # Add qubit 1
    dc.add_qubit(2) # Node 2
    dc.apply_gate_to_qubit(1, 4) # Apply H gate to qubit 1 to create |+> state
    # Add qubit 2
    dc.add_qubit(3) # Node 3 # This is qubit 2
    dc.generate_entanglement(2) # Generate entanglement between qubit 2 and new qubit 3
    # This creates a fourth qubit in the process

    assert dc.delta==1e-6
    assert dc.channel_parameters== params
    assert dc.num_qubits==4
    assert dc.qubits.keys()== {0,1,2,3} # Three qubits

    q2_state= dc.qubits[2].rho_dicts
    q3_state= dc.qubits[3].rho_dicts

    assert q2_state == q3_state

    # Test setters
    new_params=[[0.11, 0.12, 0.13],
                [0.14, 0.15, 0.16],
                [0.17, 0.18, 0.19]]
    dc.channel_parameters= new_params
    dc.delta= 1e-5

    assert dc.delta==1e-5
    assert dc.channel_parameters== new_params


    dc.action_history=[0,1,2,3]
    assert dc.action_history== [0,1,2,3]

    dc.num_qubits=5
    assert dc.num_qubits==5

    dc.num_qubits=4

    q0_state=dc.get_rho_dicts(0)
    e_rho=cb1.get_Z_basis_states()['RHO_ZERO']
    e_PQOBJ= cb1.P_QOBJ(e_rho, location=1, entanglement_vector=[0], qubit_id=0)

    assert q0_state== e_PQOBJ.rho_dicts

def test_actions_to_gates_action1_case1():
    # The qubit wants to move to an edge node that it is already located at.
    prior_action_dict = cb1.get_test_action_dict()
    action_vector = [1, # Move
                     0, # NA
                     0, # Qubit 0
                     1, # Move to node 1
                     2] # NA
    
    params= [[0.1,0.05,0.01],
             [0.9,0.9,0.9],
             [0.9,0.9,0.9]] # Different error parameters for different channels
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element is entanglement vector

    expected_state = cb1.multi_qubit_error_gate(prior_action_dict[0][0], params[0], 0, [0])
    expected_state2= cb1.multi_qubit_error_gate(expected_state, params[0], 0, [0])

    temp_p=[[0 for _ in range(3)] for _ in range(3)]
    dc=cb1.action_dict2dc(prior_action_dict,temp_p,delta=1e-6)
    dc.channel_parameters= params

    dc.move_qubit(0,1)

    q0_rho_dict=dc.get_rho_dicts(0)
    q0_location_new= dc.get_location(0)
    q0_entanglement_vector_new= dc.get_entanglement_vector(0)


    #print(q0_state)

    compare_rho_dicts(q0_rho_dict,q0_state,channel='channel_1')

    assert q0_location_new ==1
    assert q0_entanglement_vector_new == [0]

def compare_rho_dicts(big_rho_dict1, rho_dict2,channel=None):
    for i in big_rho_dict1.keys():
        for k in big_rho_dict1[i].keys():
            if i == channel:
                np.testing.assert_allclose(big_rho_dict1[i][k].full(), rho_dict2[k].full(), err_msg="Entanglement generation on already entangled qubits did not produce expected state.")
            else:
                np.testing.assert_allclose(big_rho_dict1[i][k].full(), rho_dict2['base'].full(), err_msg="Entanglement generation on already entangled qubits did not produce expected state.")

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

    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)
    dc.move_qubit(0,2)

    compare_rho_dicts(dc.get_rho_dicts(0),q0_state,channel='channel_2')
    compare_rho_dicts(dc.get_rho_dicts(1),q1_state,channel='channel_2')

    assert q0_location ==2
    assert q0_entanglement_vector == [0,1]
    assert q1_location ==0
    assert q1_entanglement_vector == [0,1]
    
def test_actions_to_gates_action1_case2():
    prior_action_dict = cb1.get_test_action_dict()
    # Add another qubit at node 0 for testing
    q4= cb1.get_perturbed_0_state()
    prior_action_dict[4] = [q4, [4], 0] # New qubit at node 0
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels

    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)

    action_vector = [1, # Move
                     0, # NA
                     4, # Qubit 4
                     1, # Move to node 1
                     2] # NA
    
    
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q4_state= new_action_dict[4][0] # First element is rho dict
    q4_location= new_action_dict[4][2] # Third element is location
    q4_entanglement_vector= new_action_dict[4][1] # second element


    
    
    dc.move_qubit(4,1)

    print(dc.get_location(4))

    compare_rho_dicts(dc.get_rho_dicts(4),q4_state,channel='channel_1')

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
    
    dc =cb1.action_dict2dc(prior_action_dict,parameters=params)

    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q1_state= new_action_dict[1][0] # First element is rho dict
    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    dc.move_qubit(1,2)

    compare_rho_dicts(dc.get_rho_dicts(1),q1_state,channel='channel_2')
    compare_rho_dicts(dc.get_rho_dicts(0),new_action_dict[0][0],channel='channel_2')

    assert dc.get_location(1) ==2
    assert q1_entanglement_vector == [0,1]

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
    
    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)

    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element


    dc.move_qubit(0,0)
    compare_rho_dicts(dc.get_rho_dicts(0),q0_state,channel='channel_1')
    assert dc.get_location(0) ==0
    assert q0_entanglement_vector == [0]

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
    
    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)
    
    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    q0_state= new_action_dict[0][0] # First element is rho dict
    q0_location= new_action_dict[0][2] # Third element is location
    q0_entanglement_vector= new_action_dict[0][1] # second element

    q1_state= new_action_dict[1][0] # First element is rho dict
    q1_location= new_action_dict[1][2] # Third element is location
    q1_entanglement_vector= new_action_dict[1][1] # second element

    dc.move_qubit(0,0)
    compare_rho_dicts(dc.get_rho_dicts(0),q0_state,channel='channel_2')
    compare_rho_dicts(dc.get_rho_dicts(1),q1_state,channel='channel_2')

    assert dc.get_location(0) ==0
    assert dc.get_location(1) ==0
    assert q0_entanglement_vector == [0,1]
    assert q1_entanglement_vector == [0,1]

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
    
    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)

    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    new_state= new_action_dict[4][0] # First element is rho dict
    new_location= new_action_dict[4][2] # Third element is location
    new_entanglement_vector= new_action_dict[4][1] # second element

    expected_state = cb1.get_perturbed_0_state()
    expected_location = 0
    expected_entanglement_vector = [4]

    dc.move_qubit(4,0)
    compare_rho_dicts(dc.get_rho_dicts(4),new_state,channel='channel_1')
    
    assert dc.get_location(4) == expected_location # Location unchanged

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
    
    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)

    new_action_dict = cb1.actions_to_gates(prior_action_dict, action_vector, params,DEBUG=True)

    new_state_q0= new_action_dict[0][0] # First element is rho dict
    new_location_q0= new_action_dict[0][2] # Third element is location
    new_entanglement_vector_q0= new_action_dict[0][1] # second element

    new_state_q1= new_action_dict[1][0] # First element is rho dict
    new_location_q1= new_action_dict[1][2] # Third element is location
    new_entanglement_vector_q1= new_action_dict[1][1] # second element


    dc.move_qubit(1,0)
    compare_rho_dicts(dc.get_rho_dicts(0),new_state_q0,channel='channel_1')
    compare_rho_dicts(dc.get_rho_dicts(1),new_state_q1,channel='channel_1')

    assert dc.get_location(0) == new_location_q0
    assert dc.get_location(1) == new_location_q1

def test_actions_to_gates_action1_case5():
    """" Test moving a qubit from an edge node to another edge node

        This is handled by first moving to the central node with one error channel,
        then moving to the target edge node with another error channel.
    """

    # Initialize prior action dict with one qubit at node 1
    prior_action_dict = cb1.get_test_action_dict()
    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    dc=cb1.action_dict2dc(prior_action_dict,parameters=params)
    dc2=cb1.action_dict2dc(prior_action_dict,parameters=params)
    dc.move_qubit(0,2) # Move to central node
    
    dc2.move_qubit(0,0) # Move to target edge node
    dc2.move_qubit(0,2) # Move to target edge node
    
    for key in dc2.qubits[0].rho_dicts.keys():
        for k in dc2.qubits[0].rho_dicts[key].keys():
            np.testing.assert_allclose(dc.qubits[0].rho_dicts[key][k].full(), dc2.qubits[0].rho_dicts[key][k].full(), rtol=1e-6, atol=1e-8)

def test_actions_to_gates_action1_case5_entangled():
    """ Test moving a qubit from an edge node to another edge node

        This is handled by first moving to the central node with one error channel,
        then moving to the target edge node with another error channel.
    """

    # Initialize prior action dict with one qubit at node 2
    prior_action_dict = cb1.get_test_action_dict_entangled()
    dc = cb1.action_dict2dc(prior_action_dict,parameters=[[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]])
    dc2 = cb1.action_dict2dc(prior_action_dict,parameters=[[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]])
    
    dc.move_qubit(0,1) # Move to node 1


    dc2.move_qubit(0,0) # Move to central node
    dc2.move_qubit(0,1) # Move to target edge node

    for key in dc2.qubits[0].rho_dicts.keys():
        for k in dc2.qubits[0].rho_dicts[key].keys():
            np.testing.assert_allclose(dc.qubits[0].rho_dicts[key][k].full(), dc2.qubits[0].rho_dicts[key][k].full(), rtol=1e-6, atol=1e-8) 

def test_measure_all():
    dc = cb1.get_test_circuit()
    assert dc.num_qubits == 4
    dc.add_qubit(0)
    dc.add_qubit(0)
    assert dc.num_qubits == 6
    dc.measure_all_qubits()

    for q_id in dc.qubits.keys():
        assert dc.get_location(q_id) != 0

    assert dc.get_location(5) == 1
    assert dc.get_location(4) == 1

    dc2=cb1.get_test_circuit()
    dc2.add_qubit(0)
    dc2.add_qubit(0)

    dc2.move_qubit(4,1)
    dc2.move_qubit(5,1)

    for q_id in dc2.qubits.keys():
        for k in dc2.qubits[q_id].rho_dicts.keys():
            for j in dc2.qubits[q_id].rho_dicts[k].keys():
                np.testing.assert_allclose(dc.qubits[q_id].rho_dicts[k][j].full(), dc2.qubits[q_id].rho_dicts[k][j].full(), rtol=1e-6, atol=1e-8)

                
# ===========================================================================
# TEST DISTRIBUTION CIRCUIT QFIM METHODS
# ===========================================================================

def test_get_qfim():

    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    delta = 1e-6

    dc=cb1.DistributionCircuit(params,delta)
    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.add_qubit(2) # Node 2

    # Apply errors to both qubits

    dc.apply_error(0,1)
    dc.apply_error(1,2)

    qfim_q1= dc.get_QFIM(0)
    qfim_q2= dc.get_QFIM(1)

    qfim_total= dc.get_all_QFIMs()


    print(qfim_q1)
    print("\n")
    print(qfim_q2)
    print("\n")
    print(qfim_total)

    np.testing.assert_allclose(qfim_total, qfim_q1+qfim_q2, rtol=1e-6, atol=1e-8)

def test_get_qfim_entangled():

    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    
    delta = 1e-6

    dc=cb1.DistributionCircuit(params,delta) # Initialize circuit
    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.apply_gate_to_qubit(0, 4) # Apply H gate to qubit 1 to create |+> state
    dc.add_qubit(2) # Node 2 -> q1
    dc.generate_entanglement(1) # Generate entanglement between qubit 1 and new qubit 2
    dc.apply_gate_to_qubit(2, 5) # Apply S gate to qubit 2 to create |R> state

    # Apply errors to both qubits

    dc.apply_error(0,1)
    dc.apply_error(1,2)

    qfim_q1= dc.get_QFIM(0)
    qfim_q2= dc.get_QFIM(1)

    qfim_q3 = dc.get_QFIM(2)

    qfim_total= dc.get_all_QFIMs()


    print(qfim_q1)
    print("\n")
    print(qfim_q2)
    print("\n")
    print(qfim_q3)
    print("\n")
    print(qfim_total)

    np.testing.assert_allclose(qfim_total, qfim_q1+qfim_q3, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(qfim_total, qfim_q1+qfim_q2, rtol=1e-6, atol=1e-8)

# ===========================================================================
# TEST DIAGONALIZATION METHODS
# ===========================================================================

def test_diagonalize():

    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for different channels
    delta = 1e-6

    dc=cb1.DistributionCircuit(params,delta)

    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.apply_gate_to_qubit(0, 4) # Apply H gate to qubit 1 to create |+> state

    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.add_qubit(2) # Node 2

    dc.apply_gate_to_qubit(1, 5) # Apply S gate to qubit 1 to create |R> state

    dc.apply_gate_to_qubit(2, 1) # Apply I gate to qubit 2 

    dc.apply_gate_to_qubit(2, 4) # Apply X gate to qubit 0

    unitary_gates= dc.gate_history

    print("Unitary Gates Applied:")
    print(unitary_gates) 

    state=dc.output_circuit_density_matrices()

    print(state)
    #dc.plot_state()

    dc.diagonalize()

    #dc.plot_state()

    state=dc.output_circuit_density_matrices()
    print("Diagonalized States:")
    print(state)

    np.testing.assert_allclose(state.T, state, rtol=1e-6, atol=1e-8)
    
    # Check that all off diagonal elements are close to zero
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if i != j:
                assert abs(state[i,j]) < 1e-6 

def test_diagonalize_complex():

    params= [[0.1,0.2,0.3],
             [0.2,0.15,0.2],
             [0.05,0.1,0.2]] # Different error parameters for
    delta = 1e-6

    dc=cb1.DistributionCircuit(params,delta)


    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.apply_gate_to_qubit(0, 4) # Apply H gate to qubit 1 to create |+> state

    dc.add_qubit(1) # Node 1 # by default in |0> state
    dc.add_qubit(2) # Node 2

    dc.generate_entanglement(1) # Generate entanglement between qubit 1 and new qubit 2

    dc.move_qubit(1,2) # Move qubit 1 to node 2

    dc.move_qubit(2,1) # Move qubit 2 to node 1


    dc.generate_entanglement(2) # Generate entanglement between qubit 2 and new qubit 3

    dc.move_qubit(1,0) # Move qubit 1 to node 0


    dc.apply_gate_to_qubit(0, 5) # Apply S gate to qubit 1 to create |R> state

    dc.apply_gate_to_qubit(2, 1) # Apply I gate to qubit 2

    dc.apply_gate_to_qubit(3, 4) # Apply X gate to qubit 3


    state1=dc.output_circuit_density_matrices()
    #dc.plot_state()    
    qfim1=dc.get_all_QFIMs()

    dc.diagonalize()
    state2=dc.output_circuit_density_matrices()
    qfim2=dc.get_all_QFIMs()
    #dc.plot_state()

    # Check that the state is diagonal
    np.testing.assert_allclose(state2.T, state2, rtol=1e-6, atol=1e-8)
    for i in range(state2.shape[0]):
        for j in range(state2.shape[1]):
            if i != j:
                assert abs(state2[i,j]) < 1e-6

    # Check that the QFIM is unchanged
    np.testing.assert_allclose(qfim1, qfim2, rtol=1e-6, atol=1e-8)

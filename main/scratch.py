import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust the path as needed
import helper_functions.qubit_mover_2 as qm2
import sympy as sp
import numpy as np


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


#nodes_to_paths([x for x in range(1,14)]


def moves_to_gates(move_list):
    state0 = qm2.get_state0()
    temp_rho=state0 # Initialize the qubit in the 0 state
    x_gate=sp.Matrix(np.matrix([[0,1],[1,0]]))
    h_gate=sp.Matrix(np.matrix([[1,1],[1,-1]]))/np.sqrt(2)
    p1,p2,p3 = sp.symbols('p1 p2 p3') # Parameter symbols
    
    for move in move_list:
        if move == 1:
            temp_rho=((1-p1)*temp_rho+p1*x_gate*temp_rho*x_gate)
        elif move == 2:
            temp_rho=((1-p2)*temp_rho+p2*x_gate*temp_rho*x_gate)
        elif move == 3:
            temp_rho=((1-p3)*temp_rho+p3*x_gate*temp_rho*x_gate)
        elif move == 4:
            temp_rho=h_gate*temp_rho*h_gate
        elif move == 0: # Identity gate (does nothing)
            continue
        else:
            #print("Invalid move detected")
            return -10

    return temp_rho

print(moves_to_gates([1,4]))
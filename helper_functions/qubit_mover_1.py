# QFIM calculations libraries
from itertools import combinations
from sympy import symbols, prod, Add, Matrix, lambdify

# Standard libraries
import numpy as np

## Helper Functions
def even_subsets(input_list):
    subsets = []
    for r in range(0, len(input_list) + 1, 2):  # Only even lengths: 0, 2, 4, ...
        subsets.extend(combinations(input_list, r))
    return [list(subset) for subset in subsets]

def list_subtract(a, b):
    result = a.copy()
    for item in b:
        if item in result:
            result.remove(item)
    return result

def symbolic_sum_of_products(full_set):
    subsets = even_subsets(full_set)
    symbols_list = symbols('p1 p2 p3')
    hat_symbols_list = symbols('p1_hat p2_hat p3_hat')

    terms = []
    for subset in subsets:
        if subset:  # skip empty subsets
            terms_list = [symbols_list[i - 1] for i in subset]
            remainder = list_subtract(full_set, subset)
            if remainder:
                terms_list += [hat_symbols_list[i - 1] for i in remainder]
            term = prod(terms_list)
            terms.append(term)
        else:
            # For the empty subset, use all hat symbols
            remainder = list_subtract(full_set, subset)
            term=prod([hat_symbols_list[i - 1] for i in remainder])
            terms.append(term)
    return Add(*terms)

def equations_to_eigen_vals(expressions):
    e1,e2,e3=expressions

    element_1=e1*e2*e3 # No bits flip

    element_2=e1*e2*(1-e3) # Bit 3 flips

    element_3=e1*(1-e2)*e3 # Bit 2 flips

    element_4=e1*(1-e2)*(1-e3) # Bits 2 and 3 flip

    element_5=(1-e1)*e2*e3 # Bit 1 flips

    element_6=(1-e1)*e2*(1-e3) # Bits 1 and 3 flip

    element_7=(1-e1)*(1-e2)*e3 # Bits 1 and 2 flip

    element_8=(1-e1)*(1-e2)*(1-e3) # Bits 1, 2, and 3 flip

    eigen_vals = [
        element_1, element_2, element_3, element_4,
        element_5, element_6, element_7, element_8
    ]

    return eigen_vals


def quantum_fisher_information(lambdas,thetas):

    symbols_list = symbols('p1 p2 p3')
    p1_hat, p2_hat, p3_hat = symbols('p1_hat p2_hat p3_hat')
    p1, p2, p3 = symbols_list        

    n = len(symbols_list)
    F = Matrix.zeros(n, n)

    subs_dict = {symbols('p1_hat'): 1 - symbols('p1'),
                 symbols('p2_hat'): 1 - symbols('p2'),
                 symbols('p3_hat'): 1 - symbols('p3')}
    lambdas_sub = [lam.subs(subs_dict) for lam in lambdas]

    for i in range(n):
        for j in range(n):
            s = 0
            for lam in lambdas_sub:
                # Avoid division by zero
                s += (1/lam) * lam.diff(symbols_list[i]) * lam.diff(symbols_list[j])
            F[i, j] = s

    number_F=F.subs({
        p1: thetas[0],
        p2: thetas[1],
        p3: thetas[2]
    })
    return number_F

def nodes_to_paths(node_list):
    path_list=[]
    path_list.append(node_list[0])

    if len(node_list)==1: # Ensures that single node lists are handled correctly
        print("ERROR: Single node list detected")
        return path_list

    for i in range(1,len(node_list)):
        path_list.append(node_list[i])
        path_list.append(node_list[i])

    filtered_list = [x for x in path_list if x != 0]
    filtered_list.pop()

    return filtered_list

def reward(node_lists,thetas):

    #################################
    ## Make sure the protocol is valid
    #################################

    # Make sure that the first element is not 0 and the last element is 0
    for node_list in node_lists:
        if node_list[0]==0 or node_list[1]==0 or node_list[-1]!=0:
            #print("Yikes")
            return -10 # Large negative reward for invalid input
        found_zero = False
        for val in node_list:
            if val == 0:
                found_zero = True
            elif found_zero and val != 0:
                #print("Invalid: nonzero after zero")
                return -10  # or any large negative reward

    ################################
    ## Calculate the QFIM
    #################################
    
    path_lists=[]
    for node_list in node_lists:
        path_list=nodes_to_paths(node_list)
        path_lists.append(path_list)

    e1=symbolic_sum_of_products(path_lists[0])
    e2=symbolic_sum_of_products(path_lists[1])
    e3=symbolic_sum_of_products(path_lists[2])

    eigen_vals=equations_to_eigen_vals([e1,e2,e3])
    F=quantum_fisher_information(eigen_vals,thetas)

    try:
        F_inv = F.inv()
        qcrb= F_inv.trace()

        max_qcrb = 1000.0

        
        reward_val=1-qcrb/max_qcrb # Higher reward for lower qcrb, normalized between 0 and 1
        reward_val = np.clip(reward_val, 0, 1)  # Ensure reward is within [0, 1]

        
    except Exception as e:
        # If F is singular or not invertible, give a large negative reward
        reward_val = -10

    return reward_val



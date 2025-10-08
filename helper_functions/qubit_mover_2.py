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


# ============================================================================
# QFIM and QCRB Calculation Functions
# ============================================================================

# Symbolic Calculations
def calculate_QFIM(rhos: list[sp.Matrix],thetas=None):
    # compute eigen values

    F=sp.zeros(len(SYMBOLS_LIST), len(SYMBOLS_LIST))
    n=len(SYMBOLS_LIST)

    symbols_list=get_all_symbols()

    for rho in rhos:

        lambdas=rho.eigenvals()


        for i in range(n):
            for j in range(n):
                s = 0
                for lam in lambdas:
                    if lam == 0:
                        continue  # Skip zero eigenvalues to avoid division by zero
                    s += (1/lam) * lam.diff(symbols_list[i]) * lam.diff(symbols_list[j])
                F[i, j] += s

    
    if thetas is not None:
        F=F.subs({p1: thetas[0], p2: thetas[1], p3: thetas[2]})
    return F

def calculate_QCRB(F: sp.Matrix):
    return F.inv().trace()


# Numeric Calculations

def calculate_eigenvals_numerical(rho: sp.Matrix, parameters: list[float]):
    """
    Calculate eigenvalues of a symbolic density matrix at specific parameter values.
    
    This is a helper function that evaluates the eigenvalues of a symbolic density matrix
    by substituting numerical parameter values and using numpy's eigenvalue solver.
    
    Args:
        rho (sp.Matrix): Symbolic density matrix containing parameters p1, p2, p3
        parameters (list[float]): Parameter values [p1_val, p2_val, p3_val] to substitute
    
    Returns:
        np.ndarray: Array of eigenvalues (real numbers for Hermitian matrices)
    
    Example:
        >>> rho = qm2.moves_to_gates([1, 4, 13])
        >>> eigenvals = calculate_eigenvals_numerical(rho, [0.1, 0.2, 0.3])
        >>> print(f"Eigenvalues: {eigenvals}")
    """
    # Get symbolic parameter variables
    symbols_data = get_all_symbols()
    p1, p2, p3 = symbols_data
    
    # Create numerical function from symbolic matrix
    
    # Evaluate matrix at given parameter values
    rho_subst = rho.subs({p1: parameters[0], p2: parameters[1], p3: parameters[2]})

    rho_num = np.array(rho_subst.evalf(), dtype=complex)
    
    # Calculate eigenvalues (use eigvals for just eigenvalues, eigh for Hermitian)
    eigenvals = np.linalg.eigvals(rho_num)
    
    # Return real parts (should be real for density matrices anyway)
    return eigenvals.real

def get_eigen_system(rho,params):
        """
        Internal helper to evaluate eigenvalues and eigenvectors at specific parameter values.
        
        Args:
            rho (sp.Matrix): Symbolic density matrix containing parameters p1, p2, p3
            params (list[float]): Parameter values [p1, p2, p3] to substitute
            
        Returns:
            tuple: (eigenvals, eigenvecs) where:
                - eigenvals: 1D array of eigenvalues
                - eigenvecs: 2D array where eigenvecs[:, i] is i-th eigenvector
        """
        # Get symbolic parameter variables from the quantum module
        symbols_data = get_all_symbols()
        p1, p2, p3 = symbols_data  # Extract [p1, p2, p3] symbols
        
        # Create a numerical function from the symbolic density matrix
        rho_subst = rho.subs({p1: params[0], p2: params[1], p3: params[2]})

        rho_num = np.array(rho_subst.evalf(), dtype=complex)
        
        # Evaluate the density matrix at the given parameter values

        # Compute eigenvalues and eigenvectors
        # Use eigh() for Hermitian matrices (guarantees real eigenvalues)
        eigenvals, eigenvecs = np.linalg.eigh(rho_num)
        
        return eigenvals, eigenvecs

def differentiate_eigenvals_numerical(rho: sp.Matrix, parameters: list[float], i: int, h: float=1e-5):
    """
    Calculate numerical derivatives of eigenvalues with respect to a specific parameter.
    
    This function computes d(λⱼ)/d(pᵢ) for all eigenvalues λⱼ of the density matrix 
    with respect to parameter pᵢ using central difference numerical differentiation.
    
    The key innovation is eigenvector matching: eigenvalues are tracked across parameter
    changes by matching their corresponding eigenvectors, ensuring we differentiate
    the same physical eigenvalue even if np.linalg.eigh() returns them in different orders.
    
    Args:
        rho (sp.Matrix): Symbolic density matrix containing parameters p1, p2, p3
        parameters (list[float]): Current parameter values [p1_val, p2_val, p3_val]
        i (int): Index of parameter to differentiate with respect to (0=p1, 1=p2, 2=p3)
        h (float, optional): Step size for numerical differentiation. Defaults to 1e-5.
                           Smaller h gives higher accuracy but may suffer from round-off errors.
    
    Returns:
        np.array: Array of eigenvalue derivatives where element j contains d(λⱼ)/d(pᵢ)
                 Shape: (n_eigenvalues,) where n_eigenvalues is typically 2 for 2x2 matrices
    
    Mathematical Formula:
        Uses central difference: d(λⱼ)/d(pᵢ) ≈ [λⱼ(pᵢ + h) - λⱼ(pᵢ - h)] / (2h)
        
    Example:
        >>> rho = qm2.moves_to_gates([1, 4, 13])  # Some quantum protocol
        >>> params = [0.1, 0.2, 0.3]  # [p1, p2, p3]
        >>> derivs = differentiate_eigenvals_numerical(rho, params, i=0)  # d/dp1
        >>> print(f"d(eigenvalues)/dp1 = {derivs}")
        d(eigenvalues)/dp1 = [0.0234, -0.0156]  # For 2x2 matrix with 2 eigenvalues
    
    Notes:
        - Assumes density matrix is Hermitian (uses np.linalg.eigh)
        - Eigenvector matching prevents eigenvalue crossing issues
        - Step size h should be small enough for accuracy but large enough to avoid
          numerical precision issues (1e-5 is usually a good compromise)
    """
    
    
    # Create perturbed parameter sets for central difference
    # Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    lower_params = parameters.copy()  # pᵢ - h
    upper_params = parameters.copy()  # pᵢ + h
    lower_params[i] -= h              # Decrease i-th parameter by h
    upper_params[i] += h              # Increase i-th parameter by h
    
    # Evaluate eigenvalue systems at perturbed parameter values
    eigenvals_lower, eigenvecs_lower = get_eigen_system(rho,lower_params)
    eigenvals_upper, eigenvecs_upper = get_eigen_system(rho,upper_params)
    
    # Match eigenvalues by eigenvector similarity to handle eigenvalue ordering
    # Problem: np.linalg.eigh() may return eigenvalues in different orders
    # Solution: Match eigenvalues based on their corresponding eigenvector similarity
    derivatives = []
    
    # Iterate through each eigenvector in the lower parameter system
    for j, vec_lower in enumerate(eigenvecs_lower.T):  # .T to iterate over columns
        """
        For each eigenvector in the lower system, find the most similar eigenvector
        in the upper system by computing overlap integrals |⟨ψᵢᵘᵖᵖᵉʳ|ψⱼˡᵒʷᵉʳ⟩|²
        """
        
        # Calculate overlap integrals between this lower eigenvector and all upper eigenvectors
        # |⟨ψᵢᵘᵖᵖᵉʳ|ψⱼˡᵒʷᵉʳ⟩|² = |eigenvecs_upper[:, i]† · vec_lower|²
        overlaps = np.abs(np.dot(eigenvecs_upper.T.conj(), vec_lower))**2
        
        # Find the upper eigenvector with maximum overlap (best match)
        # This corresponds to the same physical eigenvalue across parameter change
        best_match = np.argmax(overlaps)
        
        # Calculate numerical derivative using central difference
        # d(λⱼ)/d(pᵢ) ≈ [λⱼ(pᵢ + h) - λⱼ(pᵢ - h)] / (2h)
        derivative = (eigenvals_upper[best_match] - eigenvals_lower[j]) / (2 * h)
        derivatives.append(derivative)
    
    # Convert to numpy array for consistent return type
    # Result: derivatives[j] = d(λⱼ)/d(pᵢ) for j-th eigenvalue
    return np.array(derivatives)  # An array of derivatives for each eigenvalue with respect to parameter i

def calculate_QFIM_numerical(rhos: list[sp.Matrix], parameters: list[float],debug: bool=False):
    """
    Calculate the Quantum Fisher Information Matrix (QFIM) numerically.
    
    This function computes the QFIM using numerical differentiation of eigenvalues.
    The QFIM quantifies how much information about parameters can be extracted
    from quantum measurements on the given density matrices.
    
    Mathematical Formula:
        F[i,j] = Σₖ Σ_ρ (1/λₖ) * (dλₖ/dpᵢ) * (dλₖ/dpⱼ)
        
    Where:
        - λₖ are eigenvalues of density matrix ρ
        - dλₖ/dpᵢ is derivative of k-th eigenvalue w.r.t. parameter pᵢ
        - Sum is over all eigenvalues k and all density matrices ρ
    
    Args:
        rhos (list[sp.Matrix]): List of symbolic density matrices, one for each qubit.
                               Each matrix should contain parameters p1, p2, p3.
        parameters (list[float]): Current parameter values [p1_val, p2_val, p3_val].
    
    Returns:
        np.ndarray: 3×3 QFIM matrix where F[i,j] represents Fisher information
                   between parameters pᵢ and pⱼ. Shape: (3, 3)
    
    Raises:
        Exception: If eigenvalue calculation fails or matrices are incompatible
    
    Example:
        >>> rhos = [qm2.moves_to_gates(path) for path in path_lists]
        >>> params = [0.1, 0.2, 0.3]
        >>> F = calculate_QFIM_numerical(rhos, params)
        >>> print(f"QFIM shape: {F.shape}")  # (3, 3)
        >>> det_F = np.linalg.det(F)
        >>> print(f"QFIM determinant: {det_F}")
    
    Notes:
        - Uses differentiate_eigenvals_numerical() for robust eigenvalue derivatives
        - Filters out near-zero eigenvalues to avoid division by zero
        - Sums contributions from all density matrices in the list
    """
    
    if debug:
        print("Calculating QFIM numerically...")
        print(f"Parameters: {parameters}")
        print(f"Number of density matrices: {len(rhos)}")
    # Get the parameter symbols and determine matrix dimensions
    symbols_data = get_all_symbols()
    n_params = len(symbols_data)  # Should be 3 for [p1, p2, p3]
    
    # Initialize QFIM matrix
    F = np.zeros((n_params, n_params))
    
    
    # Process each density matrix
    for  rho in rhos:

        if debug:
            print("Processing calculating eigenvals numerically")
        # Get eigenvalues at current parameter values
        eigenvals = calculate_eigenvals_numerical(rho, parameters)
        
        # Filter out near-zero eigenvalues to avoid division by zero
        valid_mask = np.abs(eigenvals) > 1e-12
        valid_eigenvals = eigenvals[valid_mask]
                    
        # Pre-calculate all eigenvalue derivatives for efficiency
        # derivatives[i] contains d(λₖ)/dpᵢ for all eigenvalues λₖ
        all_derivatives = []
        if debug:
            print("Calculating all eigenvalue derivatives")
        for i in range(n_params):
            deriv_all = differentiate_eigenvals_numerical(rho, parameters, i)
            
            # Keep only derivatives for valid eigenvalues
            #deriv_valid = deriv_all[valid_mask] if len(deriv_all) == len(eigenvals) else deriv_all
            all_derivatives.append(deriv_all)
        
        # Calculate QFIM elements using the formula: F[i,j] = Σₖ (1/λₖ) * (dλₖ/dpᵢ) * (dλₖ/dpⱼ)
        for i in range(n_params):
            for j in range(n_params):
                contributions = (1/valid_eigenvals) * all_derivatives[i] * all_derivatives[j]
                F[i, j] += np.sum(contributions)
            
            
            
    
    return F

def calculate_QCRB_numerical(F: np.ndarray):
    """
    Calculate the Quantum Cramér-Rao Bound (QCRB) from the Quantum Fisher Information Matrix (QFIM).
    
    The QCRB provides a lower bound on the variance of unbiased estimators of parameters
    encoded in a quantum state. It is given by the inverse of the QFIM.
    
    Mathematical Formula:
        Cov(θ) ≥ tr[(F⁻¹)]
        
    Where:
        - Cov(θ) is the covariance matrix of parameter estimates
        - F is the QFIM matrix
    
    Args:
        F (np.ndarray): 3×3 QFIM matrix where F[i,j] represents Fisher information
                        between parameters pᵢ and pⱼ. Shape: (3, 3)
    
    Returns:
        np.ndarray: 3×3 QCRB matrix which is the inverse of the QFIM.
                    Shape: (3, 3)
    
    Raises:
        np.linalg.LinAlgError: If the QFIM is singular and cannot be inverted.
    
    Example:
        >>> F = np.array([[10, 2, 1], [2, 8, 0.5], [1, 0.5, 5]])
        >>> QCRB = calculate_QCRB_numerical(F)
        >>> print(f"QCRB shape: {QCRB.shape}")  # (3, 3)
        >>> print(QCRB)

    """

    
    # Calculate the inverse of the QFIM
    F_inv = np.linalg.inv(F)
    
    # Return the transpose of the inverse as the QCRB
    return np.trace(F_inv)

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
    if num_moves<3:
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

def paths_to_gates(move_list):
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
            return -1
    
    return temp_rho

# ============================================================================
# Reward Function Calculation
# ============================================================================
def reward(node_lists: list[list[int]],
           thetas:list[float],
           min_log=-2, 
           max_log=4,
           sensitivity_power=1.0/2):
    

    # Make sure there are an even number of hadamard gates between nodes # Measure in the computational basis
    path_lists=[]
    for node_list in node_lists:
        path_list=nodes_to_paths(node_list)
        if path_list == -1:
            return -1 # Invalid Move as determined by nodes_to_paths
        elif path_list.count(4) % 2 != 0:
            #print("Odd hadamards")
            return -1 # Invalid path due to odd number of hadamards wrong basis
        
        path_lists.append(path_list)
   
    ################################
    ## Calculate the QFIM
    #################################
    
    rho_1=paths_to_gates(path_lists[0])
    rho_2=paths_to_gates(path_lists[1])
    rho_3=paths_to_gates(path_lists[2])
    
    
    F=calculate_QFIM([rho_1,rho_2,rho_3],thetas=thetas) # Calculate the QFIM for the given states and parameters

    #sp.pprint(F)

    try:
        qcrb= calculate_QCRB(F)

    except Exception as e:
        # If F is singular or not invertible, give a large negative reward, this implies the protocol gives 0 information on at least one parameter
        return -1
    

    # Normalize the reward to be between 0 and 1
    # max = 10^4
    # min = 10^2

    qcrb_float = float(qcrb)  # Convert Rational to float
    normalized_reward=normalize_reward(qcrb_float, 
                                       min_log=min_log, 
                                       max_log=max_log,
                                       sensitivity_power=sensitivity_power)

    return float(normalized_reward)


def normalize_reward(qcrb: float,min_log=-2, max_log=4,sensitivity_power=1/2.0):
    # Takes in a value from around 10^-2 to 10^4 and normalizes it to a reward between 0 and 1
    # Uses a logarithmic scale to make small differences at low QCRB more significant
    # Also applies a power to increase sensitivity at high rewards (low QCRB)

    log_qcrb = np.log10(abs(qcrb))  # Convert to log scale
    normalized_reward = (max_log - log_qcrb) / (max_log - min_log)


    if normalized_reward < 0:
        enhanced_reward = -1
    else:
        enhanced_reward = normalized_reward ** sensitivity_power 

    return (enhanced_reward)

def plot_normalized_reward():
    # A function to plot the normalized reward function to help understnad its behavior
    # Create QCRB values from 10^-2 to 10^4
    qcrb_values = np.logspace(-2, 4, 1000)  # 1000 points from 10^-2 to 10^4
    
    # Calculate normalized rewards
    normalized_rewards = [normalize_reward(qcrb) for qcrb in qcrb_values]
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Linear scale
    plt.subplot(2, 2, 1)
    plt.plot(qcrb_values, normalized_rewards, 'b-', linewidth=2)
    plt.xlabel('QCRB Value')
    plt.ylabel('Normalized Reward')
    plt.title('Normalized Reward vs QCRB (Linear Scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for x-axis
    plt.subplot(2, 2, 2)
    plt.semilogx(qcrb_values, normalized_rewards, 'r-', linewidth=2)
    plt.xlabel('QCRB Value (log scale)')
    plt.ylabel('Normalized Reward')
    plt.title('Normalized Reward vs QCRB (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Show key points
    plt.subplot(2, 2, 3)
    plt.semilogx(qcrb_values, normalized_rewards, 'g-', linewidth=2)
    
    # Mark key points
    key_qcrb = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    key_rewards = [normalize_reward(q) for q in key_qcrb]
    plt.scatter(key_qcrb, key_rewards, color='red', s=100, zorder=5)
    
    # Annotate key points
    for qcrb, reward in zip(key_qcrb, key_rewards):
        plt.annotate(f'QCRB={qcrb:.0e}\nReward={reward:.2f}', 
                    xy=(qcrb, reward), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('QCRB Value (log scale)')
    plt.ylabel('Normalized Reward')
    plt.title('Key Points on Reward Function')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Reward gradient (derivative)
    plt.subplot(2, 2, 4)
    # Calculate numerical derivative
    gradient = np.gradient(normalized_rewards, qcrb_values)
    plt.semilogx(qcrb_values, gradient, 'm-', linewidth=2)
    plt.xlabel('QCRB Value (log scale)')
    plt.ylabel('Reward Gradient')
    plt.title('Reward Function Gradient')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Reward range: [{min(normalized_rewards):.3f}, {max(normalized_rewards):.3f}]")
    print(f"Reward at QCRB=0.01: {normalize_reward(0.01):.3f}")
    print(f"Reward at QCRB=1.0: {normalize_reward(1.0):.3f}")
    print(f"Reward at QCRB=100: {normalize_reward(100):.3f}")
    print(f"Reward at QCRB=10000: {normalize_reward(10000):.3f}")



    
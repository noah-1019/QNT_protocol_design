import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust the path as needed
import helper_functions.qubit_mover_2 as qm2
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def choose_random_valid_move():
    is_valid = False
    while is_valid == False:
        num_moves = [np.random.randint(3, 10),  # Between 3 and 10 moves for each qubit
                    np.random.randint(3, 10),
                    np.random.randint(3, 10)]
        

        moves=[[],[],[]]
        for _ in range(num_moves[0]-1):
            moves[0].append(np.random.randint(1, 12))

        for _ in range(num_moves[1]-1):
            moves[1].append(np.random.randint(1, 12))
        
        for _ in range(num_moves[2]-1):
            moves[2].append(np.random.randint(1, 12))
        moves[0].append(13)  # Final measurement
        moves[1].append(13)
        moves[2].append(13)

        val = qm2.reward(moves,[0.1,0.2,0.3])

        is_valid = val > -1  # Valid if reward is not -1
        #print(is_valid)
    return moves


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
    symbols_data = qm2.get_all_symbols()
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
    symbols_data = qm2.get_all_symbols()
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
        symbols_data = qm2.get_all_symbols()
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

def verify_QFIM_calc():
    
    """"
    Simple verification function to test the numerical QFIM calculation.
    
    This function sets up a basic quantum protocol, computes the QFIM using
    the numerical method, and prints the results for manual inspection.
    
    Example:
        >>> verify_QFIM_calc()
        QFIM shape: (3, 3)
        QFIM determinant: 0.00123456
    """
    # Define a simple quantum protocol path

    nodes=[1,6,11,9,13]# Simple case
    #nodes = [1, 2,5,2,6,7,9,9,9,10,10,13]  # Example moves ending with measurement
    
    # Convert path to symbolic density matrix
    paths = qm2.nodes_to_paths(nodes)
    print("paths converted")
    rho = qm2.paths_to_gates(paths)
    print("rho converted")
    
    # Define parameter values for p1, p2, p3
    params = [0.1, 0.2, 0.3]
    
    print("Calculating QFIM numerically...")
    # Calculate QFIM numerically
    F = calculate_QFIM_numerical([rho], params,debug=True)
    
    # Print results for verification
    print(f"QFIM shape: {F.shape}")  # Should be (3, 3)
    
    det_F = np.linalg.det(F)
    print(f"QFIM determinant: {det_F}")

    print(F)

    print("Symbolic Approach")

    qfim2=qm2.calculate_QFIM([rho],thetas=params)
    print(qfim2)
    



verify_QFIM_calc()



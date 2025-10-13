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
# Numpy version
STATE0_NUMPY = np.array([[1, 0], [0, 0]], dtype=complex)

# Sympy Version
STATE0 = sp.Matrix([[1, 0], [0, 0]])

# Quantum gates

# Numpy versions for numerical calculations
X_GATE_NUMPY = np.array([[0, 1], [1, 0]], dtype=complex)
H_GATE_NUMPY = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

# Sympy versions for symbolic calculations
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

# Numpy versions

def get_gates_numpy():
    """Get all quantum gate matrices as numpy arrays.
    
    Returns:
        dict: Dictionary containing all gate matrices as numpy arrays
    """
    return {
        'X': X_GATE_NUMPY.copy(),
        'H': H_GATE_NUMPY.copy(),
        'state0': STATE0_NUMPY.copy()
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


def get_x_gate_numpy():
    """Return a copy of the Pauli-X gate as a numpy array."""
    return X_GATE_NUMPY.copy()

def get_h_gate_numpy():
    """Return a copy of the Hadamard gate as a numpy array."""
    return H_GATE_NUMPY.copy()

def get_state0_numpy():
    """Return a copy of the |0><0| state as a numpy array."""
    return STATE0_NUMPY.copy()
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

# Symbolic Calculations (Purely symbolic representation)
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


# Numeric Calculations (Some symbolic representation)

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


# Direct Calculations (No symbolic representation)

def match_eigenvalues(eigenvals, eigenvecs, target_eigenvecs):
    """
    Match eigenvalues and eigenvectors to a reference set based on eigenvector similarity.
    
    This function solves the eigenvalue ordering problem that arises when computing 
    numerical derivatives of eigenvalues. When we perturb parameters and recompute 
    eigenvalues, the ordering can change arbitrarily, making derivative calculations 
    incorrect. This function ensures consistent ordering by matching eigenvectors.
    
    Mathematical Background:
    ========================
    When computing the derivative of eigenvalue λᵢ with respect to parameter θⱼ:
        
        ∂λᵢ/∂θⱼ ≈ [λᵢ(θⱼ + h) - λᵢ(θⱼ - h)] / (2h)
    
    The problem is that λᵢ(θⱼ ± h) might not correspond to the same physical 
    eigenvalue as λᵢ(θⱼ) due to arbitrary reordering by numerical solvers.
    
    Algorithm:
    ==========
    1. Use eigenvectors as "fingerprints" to identify which eigenvalues correspond
    2. For each target eigenvector, find the most similar eigenvector in the new set
    3. Match eigenvalues based on eigenvector similarity using inner products
    4. Ensure one-to-one matching (no eigenvalue is matched twice)
    
    Parameters:
    ===========
    eigenvals : np.ndarray
        Array of eigenvalues to be reordered, shape (n,)
        These are the "new" eigenvalues from the perturbed matrix
        
    eigenvecs : np.ndarray  
        Array of eigenvectors corresponding to eigenvals, shape (n, n)
        Column i contains the eigenvector for eigenvals[i]
        These are the "new" eigenvectors from the perturbed matrix
        
    target_eigenvecs : np.ndarray
        Reference eigenvectors to match against, shape (n, n)
        Column i contains the reference eigenvector for the i-th eigenvalue
        These are typically from the unperturbed matrix
    
    Returns:
    ========
    matched_eigenvals : np.ndarray
        Reordered eigenvalues, shape (n,)
        matched_eigenvals[i] corresponds to target_eigenvecs[:, i]
        
    matched_eigenvecs : np.ndarray
        Reordered eigenvectors, shape (n, n)
        matched_eigenvecs[:, i] is most similar to target_eigenvecs[:, i]
    
    Algorithm Details:
    ==================
    For each target eigenvector v_target[i]:
    
    1. Compute similarity with all new eigenvectors:
       similarity[j] = |⟨v_target[i] | v_new[j]⟩|
       
       where ⟨·|·⟩ is the complex inner product (np.vdot)
       
    2. Sort similarities in descending order
    
    3. Choose the most similar eigenvector that hasn't been used yet
    
    4. Assign the corresponding eigenvalue to matched_eigenvals[i]
    
    Similarity Measure:
    ===================
    We use |⟨v₁|v₂⟩| as the similarity metric because:
    - For identical eigenvectors: |⟨v|v⟩| = 1 (maximum similarity)
    - For orthogonal eigenvectors: |⟨v₁|v₂⟩| = 0 (minimum similarity)  
    - Phase differences don't affect matching: |⟨v|e^(iφ)v⟩| = 1
    - Works for both real and complex eigenvectors
    
    Example Usage:
    ==============
    >>> # Original matrix eigendecomposition
    >>> eigenvals_0, eigenvecs_0 = np.linalg.eigh(matrix_0)
    >>> 
    >>> # Perturbed matrix eigendecomposition (ordering might change)
    >>> eigenvals_h, eigenvecs_h = np.linalg.eigh(matrix_h)
    >>>
    >>> # Match eigenvalues to maintain consistent ordering
    >>> matched_vals, matched_vecs = match_eigenvalues(eigenvals_h, eigenvecs_h, eigenvecs_0)
    >>>
    >>> # Now we can compute derivatives safely
    >>> derivative = (matched_vals - eigenvals_0) / h
    
    Physical Interpretation:
    ========================
    In quantum mechanics, eigenvalues represent observable quantities (energies, etc.)
    and eigenvectors represent quantum states. Physical continuity demands that
    states evolve smoothly under parameter changes. This function enforces that
    continuity by tracking states through their eigenvector "fingerprints."
    
    Edge Cases:
    ===========
    - Degenerate eigenvalues: If eigenvalues are nearly equal, eigenvector matching
      becomes important for distinguishing between degenerate subspaces
    - Sign ambiguity: Eigenvectors can have arbitrary global phases (v and e^(iφ)v
      represent the same state), which is handled by taking absolute value of inner product
    
    Performance Notes:
    ==================
    - Time complexity: O(n³) where n is the number of eigenvalues
    - Space complexity: O(n²) for temporary similarity calculations
    - For large matrices, consider using approximate matching algorithms
    """
    matched_eigenvals = np.zeros_like(eigenvals)
    matched_eigenvecs = np.zeros_like(eigenvecs)
    used_indices = set()

    for i, target_vec in enumerate(target_eigenvecs.T):
        # Compute similarity between target eigenvector and all new eigenvectors
        # Use absolute value of inner product to handle phase ambiguity
        similarities = [np.abs(np.vdot(target_vec, vec)) for vec in eigenvecs.T]
        
        # Sort indices by similarity in descending order (most similar first)
        for j in np.argsort(similarities)[::-1]:  
            if j not in used_indices:
                # Assign the most similar unused eigenvector/eigenvalue pair
                matched_eigenvals[i] = eigenvals[j]
                matched_eigenvecs[:, i] = eigenvecs[:, j]
                used_indices.add(j)
                break

    return matched_eigenvals, matched_eigenvecs

def calculate_QFIM_direct_singular(rho_lists: list[np.ndarray],h=1e-5):

    

    rho= rho_lists[0][0]  # The first element is the base density matrix
    rho1low, rho1high = rho_lists[1]  # The high and low approximations for the first parameter
    rho2low, rho2high = rho_lists[2]  # The high and low approximations for the second parameter
    rho3low, rho3high = rho_lists[3]  # The high and low approximations for the third parameter

    eigenvals, eigenvecs = np.linalg.eigh(rho)

    # Calculate eigenvalues and eigenvectors for perturbed density matrices

    rho1_eigenvals_high, rho1_eigenvecs_high = np.linalg.eigh(rho1high)
    rho1_eigenvals_low, rho1_eigenvecs_low = np.linalg.eigh(rho1low)

    rho2_eigenvals_high, rho2_eigenvecs_high = np.linalg.eigh(rho2high)
    rho2_eigenvals_low, rho2_eigenvecs_low = np.linalg.eigh(rho2low)

    rho3_eigenvals_high, rho3_eigenvecs_high = np.linalg.eigh(rho3high)
    rho3_eigenvals_low, rho3_eigenvecs_low = np.linalg.eigh(rho3low)

    # Sort the eigenvalues and eigenvectors, because np.linalg.eigh does not guarantee order, we will sort the eigenvalues by closest match of corresponding eigenvectors.

    
    matched_eigenvals, matched_eigenvecs = match_eigenvalues(eigenvals, eigenvecs, eigenvecs)

    rho1_eigenvals_high, rho1_eigenvecs_high = match_eigenvalues(rho1_eigenvals_high, rho1_eigenvecs_high, eigenvecs)
    rho1_eigenvals_low, rho1_eigenvecs_low = match_eigenvalues(rho1_eigenvals_low, rho1_eigenvecs_low, eigenvecs)
    rho2_eigenvals_high, rho2_eigenvecs_high = match_eigenvalues(rho2_eigenvals_high, rho2_eigenvecs_high, eigenvecs)
    rho2_eigenvals_low, rho2_eigenvecs_low = match_eigenvalues(rho2_eigenvals_low, rho2_eigenvecs_low, eigenvecs)
    rho3_eigenvals_high, rho3_eigenvecs_high = match_eigenvalues(rho3_eigenvals_high, rho3_eigenvecs_high, eigenvecs)
    rho3_eigenvals_low, rho3_eigenvecs_low = match_eigenvalues(rho3_eigenvals_low, rho3_eigenvecs_low, eigenvecs)


    # Calculate the derivatives of eigenvalues using central difference
    derivatives=[[],[],[]]# This list will hold the derivatives of each eigenvalue with respect to each parameter
    for i in range(len(matched_eigenvals)):
        der1=(rho1_eigenvals_high[i]-rho1_eigenvals_low[i])/(2*h)
        der2=(rho2_eigenvals_high[i]-rho2_eigenvals_low[i])/(2*h)
        der3=(rho3_eigenvals_high[i]-rho3_eigenvals_low[i])/(2*h)

        derivatives[0].append(der1)
        derivatives[1].append(der2)
        derivatives[2].append(der3)

    n_params = 3 # We are looking at three parameters

    valid_eigenvals = matched_eigenvals[matched_eigenvals > 1e-10]  # Avoid division by zero

    # Calculate QFIM elements using the formula: F[i,j] = Σₖ (1/λₖ) * (dλₖ/dpᵢ) * (dλₖ/dpⱼ)
    F = np.zeros((n_params, n_params))

    for i in range(n_params):
        for j in range(n_params):
            contributions = (1/valid_eigenvals) * derivatives[i] * derivatives[j]
            F[i, j] += np.sum(contributions)
    

    
    return F


def calculate_QFIM_direct(rho_list:list[np.ndarray],h=1e-5):
    F=np.zeros((3,3))# Initialize QFIM matrix, 3 parameters so 3 x 3 matrix
    


    for rho in rho_list:
        F+=calculate_QFIM_direct_singular(rho,h)

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

def nodes_to_paths(move_list: list[int],debug: bool=False):
    # Each element in the move list is an integer from 1 to 13

    ## Encoding Scheme:
    # 1-4: Node 1 with H=0,1,2,3
    # 5-8: Node 2 with H=0,1,2,3
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
        if debug:
            print("Invalid move list: too few moves")

        return -1 # Invalid single node list
    
    if move_list[-1]!=13:
        # Must end on a measurement
        if debug:
            print("Invalid move list: does not end with measurement (13)")
        return -1 # Invalid if last move is not a 13 (end move)
    
    if any(move == 13 for move in move_list[1:-1]):  
        if debug:
            print("Invalid move list: measurement (13) in the middle of moves")
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

# Numerical and symbolic version
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

# Direct Version (Numpy only)
def apply_paths(params:list[float],move_list:list[int]): 
    x = get_x_gate_numpy() # Bit flip operator
    h=get_h_gate_numpy() # hadamard gate operator
    p1_val,p2_val,p3_val=params # The network parameters

    rho = np.array([[1, 0], [0, 0]], dtype=complex) # All qubits are initialized in the 0 state.

    for move in move_list:
            if move == 1:  # error_1
                rho = (1-p1_val) * rho + p1_val * x @ rho @ x
            elif move == 2:  # error_2  
                rho = (1-p2_val) * rho + p2_val * x @ rho @ x
            elif move == 3:  # error_3
                rho = (1-p3_val) * rho + p3_val * x @ rho @ x
            elif move == 4:  # hadamard
                rho = h @ rho @ h

    return rho

def paths_to_gates_direct(move_list: list[int], parameters: list[float],h:float=1e-5):
    """Build density matrix numerically from the start."""
    # Start with numerical |0><0|
    
    p1,p2,p3 = parameters

    rho=apply_paths([p1,p2,p3],move_list)

    rho1low=apply_paths([p1-h,p2,p3],move_list)
    rho1high=apply_paths([p1+h,p2,p3],move_list)

    rho2low=apply_paths([p1,p2-h,p3],move_list)
    rho2high=apply_paths([p1,p2+h,p3],move_list)

    rho3low=apply_paths([p1,p2,p3-h],move_list)
    rho3high=apply_paths([p1,p2,p3+h],move_list)

    rhos=[[rho],[rho1low,rho1high],[rho2low,rho2high],[rho3low,rho3high]]

    return rhos



# ============================================================================
# Reward Function Calculation
# ============================================================================

# Symbolic reward function
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
    
    rho_1=paths_to_gates(path_lists[0]) # Qubit 1
    rho_2=paths_to_gates(path_lists[1]) # Qubit 2
    rho_3=paths_to_gates(path_lists[2]) # Qubit
    
    
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
        enhanced_reward = -.9 # Almost the worst reward
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

# numeric reward function

def reward_numeric(
           node_lists: list[list[int]],
           thetas:list[float],
           min_log=-2, 
           max_log=4,
           sensitivity_power=1.0/2,
           debug: bool=False):
    

    # Make sure there are an even number of hadamard gates between nodes # Measure in the computational basis
    path_lists=[]
    for node_list in node_lists:
        path_list=nodes_to_paths(node_list,debug=debug)
        if path_list == -1:
            if debug:
                print("Invalid path detected as determined by nodes_to_paths")
            return -1 # Invalid Move as determined by nodes_to_paths
        elif path_list.count(4) % 2 != 0:
            #print("Odd hadamards")
            if debug:
                print("Odd number of hadamards detected, invalid path")
            return -1 # Invalid path due to odd number of hadamards wrong basis
        
        path_lists.append(path_list)
   
    ################################
    ## Calculate the QFIM
    #################################
    
    rho_1=paths_to_gates(path_lists[0])
    rho_2=paths_to_gates(path_lists[1])
    rho_3=paths_to_gates(path_lists[2])
    
    
    F=calculate_QFIM_numerical([rho_1,rho_2,rho_3],parameters=thetas) # Calculate the QFIM for the given states and parameters

    #sp.pprint(F)

    try:
        qcrb= calculate_QCRB_numerical(F)

    except Exception as e:
        # If F is singular or not invertible, give a large negative reward, this implies the protocol gives 0 information on at least one parameter
        if debug:
            print("Singular QFIM detected, invalid protocol")
        return -.9
    

    # Normalize the reward to be between 0 and 1
    # max = 10^4
    # min = 10^2

    qcrb_float = float(qcrb)  # Convert Rational to float
    normalized_reward=normalize_reward(qcrb_float, 
                                       min_log=min_log, 
                                       max_log=max_log,
                                       sensitivity_power=sensitivity_power)

    return float(normalized_reward)

# direct reward function

def reward_direct(
           node_lists: list[list[int]],
           thetas:list[float],
           min_log: int =-2, 
           max_log: int=4,
           sensitivity_power: float=1.0/2,
           h: float =1e-5,
           debug: bool=False):
    
    ################################
    ## Verify the move is correct
    #################################

    # Make sure there are an even number of hadamard gates between nodes # Measure in the computational basis
    path_lists=[]
    for node_list in node_lists:
        path_list=nodes_to_paths(node_list,debug=debug)
        if path_list == -1:
            if debug:
                print("Invalid path detected as determined by nodes_to_paths")
            return -1 # Invalid Move as determined by nodes_to_paths
        elif path_list.count(4) % 2 != 0:
            #print("Odd hadamards")
            if debug:
                print("Odd number of hadamards detected, invalid path")
            return -1 # Invalid path due to odd number of hadamards wrong basis
        
        path_lists.append(path_list)
   
    ################################
    ## Calculate the QFIM
    #################################
    
    rho_1=paths_to_gates_direct(path_lists[0],parameters=thetas,h=h)
    rho_2=paths_to_gates_direct(path_lists[1],parameters=thetas,h=h)
    rho_3=paths_to_gates_direct(path_lists[2],parameters=thetas,h=h)
    
    
    F=calculate_QFIM_direct([rho_1,rho_2,rho_3],h=h) # Calculate the QFIM for the given states and parameters

    #sp.pprint(F)

    try:
        qcrb= calculate_QCRB_numerical(F) # The numerical and direct versions use the same QCRB calculation

    except Exception as e:
        # If F is singular or not invertible, give a large negative reward, this implies the protocol gives 0 information on at least one parameter
        if debug:
            print("Singular QFIM detected, invalid protocol")
        return -.9
    

    # Normalize the reward to be between 0 and 1
    # max = 10^4
    # min = 10^2

    qcrb_float = float(qcrb)  # Convert Rational to float
    normalized_reward=normalize_reward(qcrb_float, 
                                       min_log=min_log, 
                                       max_log=max_log,
                                       sensitivity_power=sensitivity_power)

    return float(normalized_reward)


# ============================================================================
# Functions to observe enviornment
# ============================================================================

def choose_random_valid_move(params=[0.1,0.2,0.3]):

    moves = [[], [], []]  # Three qubits
    nodes=[1, 5, 9]  # Starting nodes for the three qubits
    for i in range(3): # Add this loop to generate moves for each qubit
        num_moves = np.random.randint(3, 5)  # Between 3 and 10 moves for each qubit
        moves[i]= [random.choice(nodes) for _ in range(num_moves)]

        # Add Hadamard gates
        h_counter=0 # Counts the number of Hadamard gates added
        for j in range(len(moves[i])-1):
            hadamard_orientation=np.random.randint(0,4) # Randomly choose one of the 4 Hadamard orientations

            if hadamard_orientation == 1 or hadamard_orientation == 2:
                h_counter+=1
            elif hadamard_orientation == 3:
                h_counter+=2


            moves[i][j]+=hadamard_orientation # Append the orientation to the node number

        if h_counter % 2 == 1: # If there's an odd number of Hadamard gates, add one more at the end
            moves[i].append(random.choice([2,6,10])) # Append a Hadamard gate at the end, each one of these moves is a hadamard gate plus a move.
            h_counter+=1

        # Ensure the last move is always a measurement
        moves[i].append(13) # Finally, append the measurement operation

        # paths=qm2.nodes_to_paths(moves[i])
        # print("*"*50)
        # print("Moves:", moves[i])
        # print("Num Hadamardds:", h_counter)
        # print("Paths: ", paths )
        # print("Other H calculation: ",paths.count(4))
        

    reward=reward_numeric(moves, thetas=params)

    

    return reward

def choose_random_valid_move_direct(params=[0.1,0.2,0.3]):

    moves = [[], [], []]  # Three qubits
    nodes=[1, 5, 9]  # Starting nodes for the three qubits
    for i in range(3): # Add this loop to generate moves for each qubit
        num_moves = np.random.randint(3, 5)  # Between 3 and 10 moves for each qubit
        moves[i]= [random.choice(nodes) for _ in range(num_moves)]

        # Add Hadamard gates
        h_counter=0 # Counts the number of Hadamard gates added
        for j in range(len(moves[i])-1):
            hadamard_orientation=np.random.randint(0,4) # Randomly choose one of the 4 Hadamard orientations

            if hadamard_orientation == 1 or hadamard_orientation == 2:
                h_counter+=1
            elif hadamard_orientation == 3:
                h_counter+=2


            moves[i][j]+=hadamard_orientation # Append the orientation to the node number

        if h_counter % 2 == 1: # If there's an odd number of Hadamard gates, add one more at the end
            moves[i].append(random.choice([2,6,10])) # Append a Hadamard gate at the end, each one of these moves is a hadamard gate plus a move.
            h_counter+=1

        # Ensure the last move is always a measurement
        moves[i].append(13) # Finally, append the measurement operation

        # paths=qm2.nodes_to_paths(moves[i])
        # print("*"*50)
        # print("Moves:", moves[i])
        # print("Num Hadamardds:", h_counter)
        # print("Paths: ", paths )
        # print("Other H calculation: ",paths.count(4))
        

    reward=reward_direct(moves, thetas=params)

    

    return reward

def plot_random_rewards_histogram(iterations,random:bool=True,params=None):


    rewards=[]

    if random==False and params is not None:
        for _ in range(iterations):
            rewards.append(choose_random_valid_move_direct(params = params))
    elif random==True:
        for _ in range(iterations):
            rewards.append(choose_random_valid_move_direct(params = np.random.uniform(0, 0.5, size=3).tolist()))
    else:
        raise ValueError("If random is False, params must be provided.")
    
    
    # Simple histogram
    plt.hist(rewards, bins=30)
    plt.xlabel('Rewards')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.show()


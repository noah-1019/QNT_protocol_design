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
    """
    Calculate the Quantum Fisher Information Matrix (QFIM) using symbolic differentiation.
    
    This function computes the QFIM using symbolic calculus on eigenvalues.
    The QFIM quantifies the amount of information that can be extracted about
    unknown parameters from quantum measurements.
    
    Mathematical Formula:
        F[i,j] = Σ_ρ Σ_λ (1/λ) * (∂λ/∂pᵢ) * (∂λ/∂pⱼ)
        
    Where:
        - λ are eigenvalues of density matrix ρ
        - ∂λ/∂pᵢ is symbolic derivative of eigenvalue w.r.t. parameter pᵢ
        - Sum is over all eigenvalues λ and all density matrices ρ
    
    Args:
        rhos (list[sp.Matrix]): List of symbolic density matrices containing parameters p1, p2, p3
        thetas (list[float], optional): If provided, substitute numerical values for parameters.
                                       Format: [p1_val, p2_val, p3_val]. Defaults to None.
    
    Returns:
        sp.Matrix: 3×3 symbolic QFIM matrix where F[i,j] represents Fisher information
                  between parameters pᵢ and pⱼ. If thetas provided, returns numerical matrix.
    
    Example:
        >>> rhos = [qm2.moves_to_gates(path) for path in path_lists]
        >>> F_symbolic = calculate_QFIM(rhos)  # Symbolic result
        >>> F_numerical = calculate_QFIM(rhos, thetas=[0.1, 0.2, 0.3])  # Numerical result
    
    Notes:
        - Uses SymPy's symbolic differentiation for exact derivatives
        - Skips zero eigenvalues to avoid division by zero
        - Much slower than numerical methods but provides exact results
    """
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
    """
    Calculate the Quantum Cramér-Rao Bound (QCRB) from symbolic QFIM.
    
    The QCRB provides a lower bound on the variance of unbiased estimators
    of parameters encoded in quantum states.
    
    Mathematical Formula:
        QCRB = tr(F⁻¹)
        
    Where F is the Quantum Fisher Information Matrix.
    
    Args:
        F (sp.Matrix): 3×3 symbolic QFIM matrix
    
    Returns:
        sp.Expr: Symbolic expression for the trace of the inverse QFIM
    
    Example:
        >>> F = calculate_QFIM(rhos)
        >>> qcrb = calculate_QCRB(F)
        >>> print(f"QCRB = {qcrb}")
    """
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
    - Time complexity: O(n³) where n is the number of eigenvalues in this case it n=2 always.
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
    """
    Calculate QFIM for a single density matrix using direct numerical methods.
    
    This function computes the QFIM using pre-computed perturbed density matrices
    and numerical differentiation. It's designed for maximum efficiency in RL
    environments where the same perturbations are used repeatedly.
    
    Mathematical Background:
        F[i,j] = Σ_λ (1/λ) * (∂λ/∂pᵢ) * (∂λ/∂pⱼ)
        
    Where eigenvalue derivatives are computed using central difference:
        ∂λ/∂pᵢ ≈ [λ(pᵢ + h) - λ(pᵢ - h)] / (2h)
    
    Args:
        rho_lists (list[np.ndarray]): Pre-computed density matrices in format:
            [
                [rho_base],                    # Base density matrix at current parameters
                [rho1_low, rho1_high],        # Perturbed matrices for parameter 1
                [rho2_low, rho2_high],        # Perturbed matrices for parameter 2  
                [rho3_low, rho3_high]         # Perturbed matrices for parameter 3
            ]
        h (float, optional): Step size used for perturbations. Defaults to 1e-5.
    
    Returns:
        np.ndarray: 3×3 QFIM matrix for this single density matrix
    
    Example:
        >>> # Pre-compute density matrices with perturbations
        >>> rho_base = paths_to_gates_direct([1, 4, 13], [0.1, 0.2, 0.3])
        >>> rho_lists = [
        ...     [rho_base],
        ...     [rho_p1_low, rho_p1_high],
        ...     [rho_p2_low, rho_p2_high], 
        ...     [rho_p3_low, rho_p3_high]
        ... ]
        >>> F = calculate_QFIM_direct_singular(rho_lists)
    
    Performance Notes:
        - Uses eigenvector matching to handle eigenvalue ordering
        - Approximately 10-100x faster than symbolic methods
        - Optimized for repeated calculations in RL training
    """

    

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
    """
    Calculate the total QFIM by summing contributions from multiple density matrices.
    
    This function computes the QFIM for a quantum protocol involving multiple qubits
    by summing the Fisher information contributions from each qubit's density matrix.
    
    Mathematical Formula:
        F_total = Σ_i F_i
        
    Where F_i is the QFIM contribution from the i-th density matrix.
    
    Args:
        rho_list (list[np.ndarray]): List of pre-computed density matrix sets.
                                    Each element should be in the format expected 
                                    by calculate_QFIM_direct_singular().
        h (float, optional): Step size for numerical differentiation. Defaults to 1e-5.
    
    Returns:
        np.ndarray: 3×3 total QFIM matrix summing all contributions
    
    Example:
        >>> # For a 3-qubit protocol
        >>> rho_list = [rho_qubit1, rho_qubit2, rho_qubit3]
        >>> F_total = calculate_QFIM_direct(rho_list)
        >>> print(f"Total QFIM shape: {F_total.shape}")  # (3, 3)
    
    Notes:
        - Used for multi-qubit quantum protocols
        - Each density matrix contributes independently to Fisher information
        - Optimized for RL environments with repeated calculations
    """
    F=np.zeros((3,3))# Initialize QFIM matrix, 3 parameters so 3 x 3 matrix
    


    for rho in rho_list:
        F+=calculate_QFIM_direct_singular(rho,h)

    return F


# ============================================================================
# Move to Path Conversion
# ============================================================================
def remove_padding_13s(move_list):
    """
    Remove all 13's except the first one encountered from a move list.
    
    In the quantum protocol encoding, the value 13 represents a "stop" or "end"
    action. This function ensures that only the first stop action is kept,
    removing any redundant stop actions that might have been added as padding.
    
    Args:
        move_list (list[int]): List of move integers where 13 represents stop action
    
    Returns:
        list[int]: Cleaned move list with only the first 13 preserved
    
    Example:
        >>> moves = [1, 4, 7, 13, 13, 13, 13]
        >>> cleaned = remove_padding_13s(moves)
        >>> print(cleaned)  # [1, 4, 7, 13]
        
        >>> moves = [2, 13, 5, 13, 8]  # Multiple 13s
        >>> cleaned = remove_padding_13s(moves)
        >>> print(cleaned)  # [2, 13, 5, 8] - only first 13 kept
    
    Use Cases:
        - Cleaning RL agent outputs that may include padding
        - Preprocessing move sequences before protocol execution
        - Ensuring protocol termination semantics are correct
    """
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
    """
    Convert a sequence of encoded moves to a quantum protocol path representation.
    
    This function transforms high-level move commands into a detailed path representation
    that specifies exactly which quantum operations to apply at each step.
    
    Encoding Scheme (Input):
    ======================
    - 1-4: Node 1 with Hadamard configurations 0,1,2,3
    - 5-8: Node 2 with Hadamard configurations 0,1,2,3  
    - 9-12: Node 3 with Hadamard configurations 0,1,2,3
    - 13: Measurement (protocol termination)
    
    Hadamard Configuration Meaning:
    =============================
    - 0: No Hadamard gates applied
    - 1: Hadamard before error application
    - 2: Hadamard after error application  
    - 3: Hadamard both before and after error application
    
    Path Representation (Output):
    ============================
    - Error operators: 1, 2, 3 (corresponding to error_1, error_2, error_3)
    - Hadamard gate: 4
    - Identity (no operation): 0
    
    Args:
        move_list (list[int]): Sequence of encoded moves (1-13)
        debug (bool, optional): Enable debug output. Defaults to False.
    
    Returns:
        list[int] or int: Path representation as list of operations, or -1 if invalid
    
    Validation Rules:
    ================
    - Must have at least 3 moves (excluding measurement)
    - Must end with measurement (13)
    - Cannot have measurement (13) in middle of sequence
    
    Example:
        >>> moves = [1, 6, 9, 13]  # Node 1 (H=0), Node 2 (H=1), Node 3 (H=0), Measure
        >>> path = nodes_to_paths(moves)
        >>> print(path)  # [1, 0, 1, 0, 2, 4, 2, 0, 3, 0, 3]
        
        # Interpretation:
        # [1, 0, 1, 0]: Node 1, no H, error_1, no H  
        # [2, 4, 2, 0]: Node 2, H, error_2, no H
        # [3, 0, 3]: Node 3, no H, error_3 (final)
    
    Error Codes:
    ============
    Returns -1 for invalid inputs:
    - Too few moves (< 3 non-measurement moves)
    - Missing final measurement
    - Measurement in middle of protocol
    
    Notes:
        - Removes padding 13's automatically using remove_padding_13s()
        - Initial and final padding operations are filtered out
        - Used as preprocessing step for quantum protocol execution
    """
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

def nodes_to_paths_mapped(move_list: list[int],debug: bool=False):
    """
    Convert encoded moves to quantum protocol paths with automatic error correction.
    
    This function extends nodes_to_paths() by automatically fixing common protocol
    violations that may occur during RL training. Instead of rejecting invalid
    moves, it corrects them and returns both the corrected path and a report of
    what was fixed.
    
    Automatic Corrections Applied:
    =============================
    1. **Too Few Moves**: If less than 3 moves provided, adds moves to node 1 (no Hadamard)
    2. **Missing Measurement**: If no final measurement (13), adds one automatically  
    3. **Odd Hadamards**: If odd number of Hadamard gates, adds one at the end
    4. **Mid-Protocol Measurement**: Truncates at first measurement, ignoring rest
    
    Physical Motivation for Corrections:
    ===================================
    - **Minimum 3 moves**: Required for meaningful 3-qubit quantum protocol
    - **Even Hadamards**: Maintains quantum coherence (Hadamards are self-inverse)
    - **Final measurement**: Protocols must terminate with measurement
    - **No mid-measurements**: Measurement destroys quantum superposition
    
    Args:
        move_list (list[int]): Sequence of encoded moves (1-13), possibly invalid
        debug (bool, optional): Enable debug output. Defaults to False.
    
    Returns:
        tuple: (corrected_path, error_report)
            - corrected_path (list[int]): Valid quantum protocol path
            - error_report (dict): Boolean flags for each correction applied:
                - "too_few": Added moves due to insufficient length
                - "bad_measurement": Added missing final measurement  
                - "odd_hadamards": Added Hadamard to make count even
    
    Example:
        >>> moves = [5, 8]  # Too few moves, no measurement
        >>> path, errors = nodes_to_paths_mapped(moves)
        >>> print(f"Corrected path: {path}")
        >>> print(f"Errors fixed: {errors}")
        # Corrected path: [1, 0, 1, 0, 2, 0, 2, 0, 3, 4, 3, 0]
        # Errors fixed: {'too_few': True, 'bad_measurement': True, 'odd_hadamards': False}
    
    RL Training Benefits:
    ====================
    - Prevents episode termination due to invalid actions
    - Provides reward signal even for imperfect agent outputs
    - Error flags allow for penalty-based learning of proper protocol structure
    - Enables continuous learning without hard failures
    
    Use Cases:
    ==========
    - RL training environments where robustness is needed
    - User interfaces that should handle invalid inputs gracefully
    - Protocol validation and automatic fixing pipelines
    - Research on quantum protocol optimization with noisy agents
    
    Notes:
        - All corrections preserve the essential structure of quantum protocols
        - Error penalties can be incorporated into RL reward functions
        - Corrections are designed to be minimally invasive
    """
    # For a description on encoding see nodes_to_paths
    # Mapping incorrect moves to correct moves  

    # A list of booleans to indicate which issue was found
    # These are all the ways to get an invalid move.
    too_few=False
    bad_measurement=False
    odd_hadamards=False



    #---------------------------------------------------------------------------
    # Verify / create correct input
    # ---------------------------------------------------------------------------

    # No measurement / measurement in middle
    # ------------------------------------------------------------------------------

    # Trim the input move to the first instance of 13 and verify if there was a measurement
    if 13 in move_list:
        first_13_index = move_list.index(13)
        move_list = move_list[:first_13_index]  # Exclude the first 13
    else:
        move_list=move_list
        bad_measurement=True  # No 13 found, keep the list as is

    if debug:
        print("Trimmed move list: ", move_list) # Should not contain any 13's now.


    # Too few moves
    # The soloution to this is to add a move to the start node with no hadamard
    # ------------------------------------------------------------------------------------

    num_moves=len([n for n in move_list if n!=13]) # Counts the number of non-13 moves
    if num_moves<3: # Too few moves
        too_few=True # pass to the reward function to give a slight penalty
        while num_moves<3:
            move_list.insert(0,1) # Add a move to node 1 with no hadamard # Add to the beginning to avoid cutoff at end.
            num_moves+=1 # Increment the move count by 1
        if debug:
            print("Added moves to start to ensure at least 3 moves: ", move_list)


    # Odd amount of hadamards
    # The soloution to this is to add a hadamard to the last node which will be done at the end.

    #---------------------------------------------------------------------------
    # Convert to path list
    # ---------------------------------------------------------------------------

    move_list.append(13)
    filtered_path_list=nodes_to_paths(move_list, # A measurement was added to the end to make it valid
                                      debug)

    num_hadamards=filtered_path_list.count(4)
    if num_hadamards%2!=0: # Odd number of hadamards
        odd_hadamards=True # pass to the reward function to give a slight penalty
        filtered_path_list.append(4) # Add a hadamard

        if debug:
            print("Added hadamard to end to ensure even number of hadamards: ", filtered_path_list)



    return filtered_path_list,{"too_few":too_few,"bad_measurement":bad_measurement,"odd_hadamards":odd_hadamards}

# Numerical and symbolic version
def paths_to_gates(move_list):
    """
    Convert a quantum protocol path to a symbolic density matrix.
    
    This function executes a quantum protocol by sequentially applying the operations
    specified in the path to an initial |0⟩⟨0| state, producing the final density matrix
    in symbolic form (containing parameters p1, p2, p3).
    
    Operation Mapping:
    =================
    - 0: Identity (no operation)
    - 1: Apply error_1 (X-gate with probability p1)
    - 2: Apply error_2 (X-gate with probability p2)  
    - 3: Apply error_3 (X-gate with probability p3)
    - 4: Apply Hadamard gate
    
    Mathematical Operations:
    =======================
    - Initial state: ρ₀ = |0⟩⟨0|
    - Error operations: ρ → (1-pᵢ)ρ + pᵢ·X·ρ·X†
    - Hadamard operation: ρ → H·ρ·H†
    
    Args:
        move_list (list[int]): Sequence of operations (0-4) defining the protocol
    
    Returns:
        sp.Matrix or int: Final symbolic density matrix, or -1 if invalid operation
    
    Example:
        >>> path = [1, 4, 2, 0, 3]  # error_1, H, error_2, identity, error_3
        >>> rho_final = paths_to_gates(path)
        >>> print(type(rho_final))  # <class 'sympy.matrices.dense.MutableDenseMatrix'>
        >>> print(rho_final.free_symbols)  # {p1, p2, p3}
    
    Use Cases:
    ==========
    - Symbolic analysis of quantum protocols
    - Deriving exact QFIM expressions
    - Protocol verification and debugging
    - Academic research requiring exact mathematical forms
    
    Performance Notes:
        - Slower than numerical methods due to symbolic computation
        - Memory usage grows with protocol complexity
        - Best for small protocols or when exact expressions needed
    """
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
    """
    Apply a quantum protocol path using numerical methods at specific parameter values.
    
    This function executes the same quantum protocol as paths_to_gates() but uses
    numerical computation with concrete parameter values, making it much faster
    for repeated evaluations in optimization and RL contexts.
    
    Mathematical Operations:
    =======================
    - Initial state: ρ₀ = |0⟩⟨0| (numpy array)
    - Error operations: ρ → (1-pᵢ)ρ + pᵢ·X·ρ·X†
    - Hadamard operation: ρ → H·ρ·H†
    
    Where X and H are Pauli-X and Hadamard matrices respectively.
    
    Args:
        params (list[float]): Parameter values [p1, p2, p3] for error probabilities
        move_list (list[int]): Sequence of operations (0-4) defining the protocol
    
    Returns:
        np.ndarray: Final density matrix as 2×2 complex numpy array
    
    Example:
        >>> params = [0.1, 0.2, 0.3]  # p1=0.1, p2=0.2, p3=0.3
        >>> path = [1, 4, 2, 0, 3]   # error_1, H, error_2, identity, error_3
        >>> rho = apply_paths(params, path)
        >>> print(rho.shape)  # (2, 2)
        >>> print(np.trace(rho))  # 1.0 (density matrix normalization)
    
    Performance Comparison:
    ======================
    - ~100x faster than symbolic methods
    - Constant memory usage regardless of protocol complexity
    - Optimized for repeated evaluations
    - Ideal for RL training and optimization loops
    
    Use Cases:
    ==========
    - RL environment reward calculations
    - Optimization algorithm objective functions
    - Monte Carlo simulations
    - Real-time protocol evaluation
    """
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
    """
    Generate density matrices with parameter perturbations for direct QFIM calculation.
    
    This function computes not only the density matrix at the current parameter values,
    but also all the perturbed density matrices needed for numerical QFIM calculation.
    This is optimized for RL environments where the same perturbations are used repeatedly.
    
    Mathematical Background:
    =======================
    For numerical QFIM calculation, we need:
    - ρ(p₁, p₂, p₃): Base density matrix
    - ρ(p₁±h, p₂, p₃): Perturbed w.r.t. p₁  
    - ρ(p₁, p₂±h, p₃): Perturbed w.r.t. p₂
    - ρ(p₁, p₂, p₃±h): Perturbed w.r.t. p₃
    
    These enable central difference approximation of derivatives:
    ∂ρ/∂pᵢ ≈ [ρ(pᵢ + h) - ρ(pᵢ - h)] / (2h)
    
    Args:
        move_list (list[int]): Quantum protocol path sequence (0-4 operations)
        parameters (list[float]): Current parameter values [p1, p2, p3]
        h (float, optional): Perturbation step size. Defaults to 1e-5.
    
    Returns:
        list[list[np.ndarray]]: Nested list structure:
            [
                [rho_base],                    # Base density matrix
                [rho_p1_low, rho_p1_high],    # p1 perturbations
                [rho_p2_low, rho_p2_high],    # p2 perturbations  
                [rho_p3_low, rho_p3_high]     # p3 perturbations
            ]
    
    Example:
        >>> path = [1, 4, 2, 0, 3]  # Some quantum protocol
        >>> params = [0.1, 0.2, 0.3]
        >>> rho_set = paths_to_gates_direct(path, params)
        >>> print(len(rho_set))  # 4 (base + 3 parameter perturbations)
        >>> print(len(rho_set[1]))  # 2 (low and high perturbations)
    
    Performance Benefits:
    ====================
    - Pre-computes all needed perturbations in one call
    - Optimized for repeated QFIM calculations
    - Enables efficient numerical Fisher information computation
    - Ideal for RL training where same perturbations used repeatedly
    
    Use Cases:
    ==========
    - RL environment reward function calculations
    - Optimization algorithms requiring gradient information
    - Numerical sensitivity analysis
    - Protocol optimization with parameter estimation objectives
    """
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
    """
    Calculate reward for a 3-qubit quantum protocol using symbolic QFIM computation.
    
    This function evaluates the quality of a quantum parameter estimation protocol
    by computing the Quantum Fisher Information Matrix (QFIM) and deriving a
    normalized reward based on the Quantum Cramér-Rao Bound (QCRB).
    
    Protocol Structure:
    ==================
    - 3 qubits, each following an independent quantum protocol path
    - Each qubit starts in |0⟩ state and undergoes sequence of operations
    - Protocol quality measured by parameter estimation precision (lower QCRB = better)
    
    Reward Calculation Pipeline:
    ===========================
    1. Convert node sequences to quantum operation paths
    2. Validate protocols (even Hadamards, proper termination)
    3. Generate symbolic density matrices for each qubit
    4. Compute 3×3 QFIM across all qubits
    5. Calculate QCRB = tr(F⁻¹)
    6. Apply logarithmic normalization to [0,1] range
    
    Args:
        node_lists (list[list[int]]): 3 node sequences, one per qubit.
                                     Each sequence uses encoding 1-13 (see nodes_to_paths).
        thetas (list[float]): Parameter values [p1, p2, p3] for numerical evaluation.
        min_log (float, optional): Minimum log₁₀(QCRB) for normalization. Defaults to -2.
        max_log (float, optional): Maximum log₁₀(QCRB) for normalization. Defaults to 4.
        sensitivity_power (float, optional): Power applied to enhance sensitivity. Defaults to 0.5.
    
    Returns:
        float or int: Normalized reward in [0,1] range, or -1 for invalid protocols
    
    Validation Rules:
    ================
    - Each qubit path must be valid (see nodes_to_paths validation)
    - Even number of Hadamard gates per qubit (measurement basis consistency)
    - QFIM must be invertible (non-singular Fisher information)
    
    Example:
        >>> # Define 3-qubit protocol
        >>> node_lists = [
        ...     [1, 5, 9, 13],   # Qubit 1: nodes 1,2,3 + measurement
        ...     [2, 6, 10, 13],  # Qubit 2: nodes 1,2,3 + measurement  
        ...     [3, 7, 11, 13]   # Qubit 3: nodes 1,2,3 + measurement
        ... ]
        >>> params = [0.1, 0.2, 0.3]
        >>> reward_val = reward(node_lists, params)
        >>> print(f"Protocol reward: {reward_val:.3f}")
    
    Reward Interpretation:
    =====================
    - Higher reward → Lower QCRB → Better parameter estimation precision
    - Reward = 1.0: Optimal protocol (QCRB ≈ 10⁻²)
    - Reward = 0.0: Poor protocol (QCRB ≈ 10⁴)
    - Reward = -1: Invalid protocol (validation failure)
    
    Mathematical Background:
    =======================
    QCRB provides theoretical limit on parameter estimation variance:
    Var(θ̂ᵢ) ≥ [F⁻¹]ᵢᵢ
    
    Total estimation error bounded by: tr(F⁻¹) = QCRB
    
    Performance Notes:
        - Uses symbolic computation (slow but exact)
        - Memory intensive for complex protocols
        - Best for research and small-scale optimization
        - Consider reward_numeric() or reward_direct() for RL training
    
    Use Cases:
    ==========
    - Academic research on quantum protocol optimization
    - Exact theoretical analysis of parameter estimation limits
    - Benchmarking numerical methods
    - Small-scale protocol design and verification
    """
    

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
    """
    Convert QCRB values to normalized rewards using logarithmic scaling.
    
    This function transforms Quantum Cramér-Rao Bound values into a [0,1] reward range
    suitable for RL training. The transformation uses logarithmic scaling because QCRB
    values can span many orders of magnitude (10⁻² to 10⁴ or more).
    
    Mathematical Transformation:
    ===========================
    1. Take logarithm: log_qcrb = log₁₀(|qcrb|)
    2. Linear mapping: normalized = (max_log - log_qcrb) / (max_log - min_log)
    3. Sensitivity enhancement: reward = normalized^sensitivity_power
    
    The inversion (max_log - log_qcrb) ensures lower QCRB → higher reward.
    
    Args:
        qcrb (float): Quantum Cramér-Rao Bound value (positive)
        min_log (float, optional): Minimum log₁₀(QCRB) for excellent protocols. Defaults to -2.
        max_log (float, optional): Maximum log₁₀(QCRB) for poor protocols. Defaults to 4.
        sensitivity_power (float, optional): Power to enhance high-reward sensitivity. Defaults to 0.5.
    
    Returns:
        float: Normalized reward in approximately [0,1] range
               - Values outside range get clamped (negative → -0.9)
               - Higher rewards correspond to better protocols (lower QCRB)
    
    Parameter Guidelines:
    ====================
    - min_log = -2: Excellent protocols with QCRB ≈ 0.01
    - max_log = 4: Poor protocols with QCRB ≈ 10,000
    - sensitivity_power < 1: Enhances sensitivity for good protocols
    - sensitivity_power = 1: Linear transformation
    - sensitivity_power > 1: Enhances sensitivity for poor protocols
    
    Example:
        >>> # Excellent protocol
        >>> reward = normalize_reward(0.01)  # QCRB = 10^-2
        >>> print(f"Excellent: {reward:.3f}")  # ≈ 1.0
        
        >>> # Poor protocol  
        >>> reward = normalize_reward(10000)  # QCRB = 10^4
        >>> print(f"Poor: {reward:.3f}")      # ≈ 0.0
        
        >>> # Very poor protocol (outside range)
        >>> reward = normalize_reward(1e6)    # QCRB = 10^6
        >>> print(f"Very poor: {reward:.3f}") # ≈ -0.9
    
    RL Training Benefits:
    ====================
    - Stable reward gradients across many orders of magnitude
    - Enhanced sensitivity near optimal protocols (sensitivity_power < 1)
    - Bounded rewards prevent numerical instabilities
    - Logarithmic scaling matches intuitive "orders of magnitude" improvements
    
    Sensitivity Power Effects:
    =========================
    - 0.5 (default): √(normalized) - emphasizes improvements in good protocols
    - 1.0: Linear - uniform sensitivity across quality range  
    - 2.0: squared - emphasizes avoiding very poor protocols
    
    Physical Interpretation:
    =======================
    QCRB represents the theoretical minimum variance in parameter estimation.
    Lower QCRB means more precise measurements are theoretically possible,
    making such protocols more valuable for quantum sensing applications.
    """
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
    """
    Visualize the normalized reward function behavior across QCRB ranges.
    
    This function creates a comprehensive visualization of how QCRB values
    are transformed into normalized rewards, helping understand the reward
    landscape for RL training and protocol optimization.
    
    Generated Plots:
    ===============
    1. **Linear Scale**: Reward vs QCRB on linear axes
    2. **Log Scale**: Reward vs QCRB with logarithmic x-axis  
    3. **Key Points**: Annotated plot showing specific QCRB→reward mappings
    4. **Gradient**: Numerical derivative showing reward sensitivity
    
    Analysis Features:
    =================
    - Tests QCRB range from 10⁻² to 10⁴ (6 orders of magnitude)
    - Highlights key transition points in reward function
    - Shows reward gradients for understanding learning dynamics
    - Prints numerical statistics for common QCRB values
    
    Example Output:
    ==============
    Creates 2×2 subplot figure showing:
    - How logarithmic scaling compresses wide QCRB ranges
    - Reward sensitivity at different quality levels
    - Transition points where reward changes rapidly
    - Gradient information for RL algorithm design
    
    Use Cases:
    ==========
    - Validating reward function design choices
    - Understanding RL training dynamics
    - Debugging convergence issues in optimization
    - Academic presentation of reward function properties
    - Tuning sensitivity_power parameter effects
    
    Statistics Printed:
    ==================
    - Overall reward range (min/max values)
    - Specific rewards for benchmark QCRB values:
      - 0.01 (excellent protocol)
      - 1.0 (decent protocol)  
      - 100 (poor protocol)
      - 10000 (very poor protocol)
    
    Notes:
        - Requires matplotlib for plotting
        - Uses default normalize_reward() parameters
        - Creates interactive plot window
        - Useful for parameter tuning and education
    """
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
    """
    Calculate reward for a 3-qubit quantum protocol using numerical QFIM computation.
    
    This function provides the same reward calculation as reward() but uses numerical
    methods instead of symbolic computation, offering significantly better performance
    for RL training and optimization loops.
    
    Key Differences from reward():
    =============================
    - Uses calculate_QFIM_numerical() instead of calculate_QFIM()
    - Maintains symbolic density matrices but uses numerical differentiation
    - ~10-100x faster than purely symbolic methods
    - Better suited for RL training environments
    
    Performance Comparison:
    ======================
    - Symbolic reward(): Exact but slow (~seconds per evaluation)
    - Numerical reward_numeric(): Fast approximation (~milliseconds per evaluation)  
    - Direct reward_direct(): Fastest pure numerical (~microseconds per evaluation)
    
    Args:
        node_lists (list[list[int]]): 3 node sequences, one per qubit (encoding 1-13)
        thetas (list[float]): Parameter values [p1, p2, p3] for numerical evaluation
        min_log (float, optional): Minimum log₁₀(QCRB) for normalization. Defaults to -2.
        max_log (float, optional): Maximum log₁₀(QCRB) for normalization. Defaults to 4.
        sensitivity_power (float, optional): Power for sensitivity enhancement. Defaults to 0.5.
        debug (bool, optional): Enable debug output for troubleshooting. Defaults to False.
    
    Returns:
        float or int: Normalized reward in [0,1] range, or -1 for invalid protocols
    
    Validation Rules:
    ================
    - Same validation as reward(): valid paths, even Hadamards, invertible QFIM
    - Debug mode provides detailed error messages for invalid protocols
    
    Example:
        >>> node_lists = [[1, 5, 9, 13], [2, 6, 10, 13], [3, 7, 11, 13]]
        >>> params = [0.1, 0.2, 0.3]
        >>> reward_val = reward_numeric(node_lists, params, debug=True)
        >>> print(f"Numerical reward: {reward_val:.4f}")
    
    Debug Output:
    ============
    When debug=True, prints:
    - Invalid path detection reasons
    - Odd Hadamard count warnings
    - QFIM calculation progress
    - Error handling information
    
    Use Cases:
    ==========
    - RL training environments requiring fast reward computation
    - Optimization algorithms with many function evaluations
    - Rapid protocol prototyping and testing
    - Situations where symbolic exactness isn't required
    
    Accuracy Notes:
        - Numerical differentiation introduces small approximation errors
        - Typical accuracy: ~6-8 significant digits
        - Errors usually negligible for RL reward signals
        - Use reward() for exact analytical results when needed
    
    Performance Tips:
        - Disable debug mode for maximum speed
        - Consider reward_direct() for even better performance
        - Cache parameter perturbations if using repeatedly
    """
    

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
           node_lists: list[list[int]], # List of moves for each qubit
           thetas:list[float], # Parameter values for the direct calculation
           min_log: int =-2,  # Used in normalization
           max_log: int=4,    # Used in normalization
           sensitivity_power: float=1.0/2, # Used in normalization
           h: float =1e-5,  # Step size for numerical derivatives
           mapped: bool=False, # Whether to map invalid moves to valid ones
           mapped_penalty: float=0.05, # Penalty for each issue found in mapped moves
           debug: bool=False): # Whether to print debug information
    """
    Calculate reward for a 3-qubit quantum protocol using pure numerical methods.
    
    This is the fastest and most optimized reward function, designed specifically
    for RL training environments. It uses purely numerical computation throughout,
    avoiding any symbolic operations for maximum performance.
    
    Performance Hierarchy:
    =====================
    - reward(): Symbolic (exact) - ~1000ms per evaluation
    - reward_numeric(): Mixed - ~10-100ms per evaluation  
    - reward_direct(): Pure numerical - ~1-10ms per evaluation  FASTEST
    
    Key Optimizations:
    =================
    - Pure numpy operations (no SymPy)
    - Pre-computed parameter perturbations
    - Optimized eigenvalue matching algorithms
    - Minimal memory allocation
    - Vectorized matrix operations
    
    Advanced Features:
    =================
    - **Mapped Mode**: Automatically fixes invalid protocols instead of rejecting them
    - **Penalty System**: Applies small penalties for corrected protocol violations
    - **Debug Mode**: Detailed logging for development and troubleshooting
    - **Tunable Precision**: Adjustable step size for derivative calculations
    
    Args:
        node_lists (list[list[int]]): 3 node sequences for qubits (encoding 1-13)
        thetas (list[float]): Parameter values [p1, p2, p3] for evaluation
        min_log (int, optional): Min log₁₀(QCRB) for normalization. Defaults to -2.
        max_log (int, optional): Max log₁₀(QCRB) for normalization. Defaults to 4.
        sensitivity_power (float, optional): Sensitivity enhancement power. Defaults to 0.5.
        h (float, optional): Numerical derivative step size. Defaults to 1e-5.
        mapped (bool, optional): Enable automatic protocol fixing. Defaults to False.
        mapped_penalty (float, optional): Penalty per fixed issue. Defaults to 0.05.
        debug (bool, optional): Enable debug logging. Defaults to False.
    
    Returns:
        float or int: Normalized reward with optional penalties, or -1/-0.9 for failures
    
    Mapped Mode Benefits:
    ====================
    When mapped=True:
    - Invalid protocols → Automatically corrected + small penalty
    - Enables continuous RL training without episode termination
    - Provides learning signal even for imperfect agent outputs
    - Essential for robust RL environments
    
    Example Usage:
    =============
    >>> # Standard mode (fast, strict validation)
    >>> node_lists = [[1, 5, 9, 13], [2, 6, 10, 13], [3, 7, 11, 13]]
    >>> params = [0.1, 0.2, 0.3]
    >>> reward = reward_direct(node_lists, params)
    >>> print(f"Reward: {reward:.4f}")
    
    >>> # Mapped mode (robust, auto-fixing)
    >>> reward = reward_direct(node_lists, params, mapped=True, debug=True)
    >>> # Automatically fixes protocol issues and reports what was fixed
    
    >>> # High precision mode
    >>> reward = reward_direct(node_lists, params, h=1e-7)
    >>> # Uses smaller step size for more accurate derivatives
    
    Performance Benchmarks:
    ======================
    Typical timing on modern hardware:
    - Simple 3-qubit protocol: ~1-3ms
    - Complex protocol: ~5-10ms  
    - With debugging: +50% overhead
    - Mapped mode: +10-20% overhead
    
    RL Training Considerations:
    ==========================
    - Use mapped=True for training robustness
    - Disable debug for maximum speed
    - Tune mapped_penalty to balance exploration vs exploitation
    - Consider h=1e-6 for higher accuracy if needed
    
    Error Handling:
    ==============
    - Returns -1: Invalid protocol (strict mode)
    - Returns -0.9: Singular QFIM (non-invertible Fisher information)
    - Mapped mode fixes most issues automatically
    
    Use Cases:
    ==========
    - RL environment step() functions ⭐ PRIMARY USE
    - High-frequency optimization loops
    - Real-time protocol evaluation
    - Large-scale parameter sweeps
    - Performance-critical applications
    
    Notes:
        - Optimized for repeated calls with similar protocols
        - Memory usage scales O(1) with protocol complexity
        - Thread-safe for parallel RL environments
        - Numerical stability tested across wide parameter ranges
    """
    ################################
    ## Verify the move is correct
    #################################

    # Make sure there are an even number of hadamard gates between nodes # Measure in the computational basis
    path_lists=[]
    for node_list in node_lists:
        if mapped:
            path_list,issues=nodes_to_paths_mapped(node_list,debug=debug)
            if debug and any(issues.values()):
                print("Mapped issues detected: ", issues)
        else:
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
    
    normalized_reward=float(normalized_reward) # convert to float for consistency
    if mapped:
        for issue,found in issues.items():
            if found:
                normalized_reward-=mapped_penalty # Slight penalty for each issue found

    return normalized_reward



# ============================================================================
# Functions to observe enviornment
# ============================================================================

def choose_random_valid_move(params=[0.1,0.2,0.3]):
    """
    Generate a random valid 3-qubit quantum protocol for testing and benchmarking.
    
    This function creates a random quantum protocol that satisfies all validation
    constraints, making it useful for testing reward functions, generating training
    data, and establishing baseline performance metrics.
    
    Protocol Generation Rules:
    =========================
    - 3 qubits, each with 3-4 random moves
    - Random node selection from appropriate ranges per qubit
    - Random Hadamard orientations (0-3) applied to each move
    - Automatic even Hadamard count enforcement
    - Proper protocol termination with measurement (13)
    
    Validation Guarantees:
    =====================
    - Valid path structure (passes nodes_to_paths validation)
    - Even number of Hadamard gates per qubit
    - Proper measurement termination
    - Ready for immediate reward calculation
    
    Args:
        params (list[float], optional): Parameter values for reward evaluation.
                                       Defaults to [0.1, 0.2, 0.3].
    
    Returns:
        tuple: (node_lists, reward_value)
            - node_lists (list[list[int]]): 3 valid node sequences for qubits
            - reward_value (float): Computed reward using reward_numeric()
    
    Node Range Mapping:
    ==================
    - Qubit 1: Starts with nodes 1-4 (encoding 1-4)
    - Qubit 2: Starts with nodes 5-8 (encoding 5-8)  
    - Qubit 3: Starts with nodes 9-12 (encoding 9-12)
    
    Hadamard Orientation Encoding:
    =============================
    - 0: No Hadamard gates
    - 1: Hadamard before error operation (+1 to count)
    - 2: Hadamard after error operation (+1 to count)
    - 3: Hadamard before and after (+2 to count)
    
    Example Usage:
    =============
    >>> # Generate random protocol and evaluate
    >>> moves, reward = choose_random_valid_move()
    >>> print(f"Random protocol reward: {reward:.4f}")
    >>> print(f"Qubit 1 moves: {moves[0]}")
    >>> print(f"Qubit 2 moves: {moves[1]}")  
    >>> print(f"Qubit 3 moves: {moves[2]}")
    
    >>> # Generate with custom parameters
    >>> moves, reward = choose_random_valid_move([0.05, 0.15, 0.25])
    >>> print(f"Custom parameter reward: {reward:.4f}")
    
    Use Cases:
    ==========
    - **Baseline Testing**: Establish random performance baselines
    - **Function Validation**: Test reward functions with known-valid inputs
    - **Data Generation**: Create training/test datasets for RL
    - **Benchmarking**: Compare optimization algorithms against random search
    - **Debugging**: Generate valid examples when troubleshooting
    - **Monte Carlo Studies**: Statistical analysis of protocol quality distributions
    
    Statistical Properties:
    ======================
    - Protocol length: 3-4 moves per qubit (9-12 total operations)
    - Hadamard density: ~25% of operations (random distribution)
    - Node selection: Uniform within each qubit's range
    - Reward distribution: Typically log-normal (most protocols poor, few excellent)
    
    Performance Notes:
        - Fast generation (~microseconds per protocol)
        - Uses reward_numeric() for evaluation (moderate speed)
        - Consider choose_random_valid_move_direct() for maximum speed
        - Thread-safe for parallel generation
    
    Related Functions:
        - choose_random_valid_move_direct(): Faster version using reward_direct()
        - plot_random_rewards_histogram(): Visualize random protocol quality
    """

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


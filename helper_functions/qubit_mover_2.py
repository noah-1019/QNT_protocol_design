"""
Qubit mover 2 helper functions. This is a file containg helper functions for the qubit mover 2 reinforcement learning agent.

- Author: Noah Plant
- Date: September 2025

Functions:
- 

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



def error_1(rho):
    state0=sp.Matrix(np.matrix([[1,0],[0,0]]))
    x_gate=sp.Matrix(np.matrix([[0,1],[1,0]]))
    h_gate=sp.Matrix(np.matrix([[1,1],[1,-1]]))/np.sqrt(2)
    p1,p2,p3 = sp.symbols('p1 p2 p3') # Parameter symbols
    symbols_list = [p1,p2,p3]
    p1_hat,p2_hat,p3_hat = sp.symbols('p1_hat p2_hat p3_hat')
    return(p1_hat*rho+p1*x_gate*rho*x_gate)

def error_2(rho):
    return(p2_hat*rho+p2*x_gate*rho*x_gate)

def error_3(rho):
    return(p3_hat*rho+p3*x_gate*rho*x_gate)

def hadamard(rho):
    return(h_gate*rho*h_gate)
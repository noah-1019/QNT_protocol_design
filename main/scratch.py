import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adjust the path as needed
import helper_functions.qubit_mover_2 as qm2
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy.typing 
import time





# Fix 1: Provide 3 separate qubit paths
print(qm2.reward_direct([
    [1,5,9,9,1,1,9,13,0,0,0,0],      # Qubit 1 path
    [1,5,9,9,1,1,9,13,0,0,0,0],   # Qubit 2 path  
    [1,5,9,9,1,1,9,13,0,0,0,0]   # Qubit 3 path
], [0.1,0.1,0.1], debug=True))
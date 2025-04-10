from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from tqdm.notebook import tqdm
from typing import List, Tuple

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model_ode(times, init: List[float], params: List[float]) -> List[float]:
    """
    Define the SIR model here
    Args:
        times: time points (not used)
        init: array of initial fractions [S_vax, S_unvax, I_vax, I_unvax, R_vax, R_unvax]
        params: array of parameters [beta, gamma, rho]

    Returns:
        [dS_vaxdt, dS_unvaxdt, dI_vaxdt, dI_unvaxdt, dR_vaxdt, dR_unvaxdt]
    """
    # CODE HERE
    S_vax = init[0]
    S_unvax = init[1]
    I_vax = init[2]
    I_unvax = init[3]
    R_vax = init[4]
    R_unvax = init[5]
    beta = params[0]
    gamma = params[1]
    rho = params[2]
    
    # 2
    dS_vaxdt = -beta * (1 - rho) * S_vax * (I_unvax + I_vax)
    dI_vaxdt = (beta * (1 - rho) * S_vax * (I_unvax + I_vax)) - (gamma * (1 - rho) * I_vax)
    dR_vaxdt = gamma * (1 - rho) * I_vax
    
    # 1
    dS_unvaxdt = -beta * S_unvax * (I_unvax + I_vax)
    dI_unvaxdt = (beta * S_unvax * (I_unvax + I_vax)) - (gamma * I_unvax)
    dR_unvaxdt = gamma * I_unvax
     
    dX_dt = [dS_vaxdt, dS_unvaxdt, dI_vaxdt, dI_unvaxdt, dR_vaxdt, dR_unvaxdt]
    
    return dX_dt

def solve_ode(init, params, t_max, t_step):
    """
    Solve the ODE model
    """
    times = np.arange(0, t_max, t_step)
    sol = solve_ivp(
        lambda t, y: model_ode(t, y, params), [0, t_max], init, t_eval=times
    )
    return sol.t, sol.y

def stochastic_model_oracle(init: List[float]) -> float:
    """
    Define the stochastic model here
    Args:
        init: array of initial fractions [S_vax, S_unvax, I_vax, I_unvax, R_vax, R_unvax]
    Returns:
        r_final: final fraction of recovered individuals
    """
    S_vax, S_unvax, I_vax, I_unvax, R_vax, R_unvax = init
    # CODE HERE
    # First sample the parameters from distributions
    beta = np.random.uniform(0.05, 0.15)
    gamma = np.random.uniform(0.005, 0.015)
    rho = np.random.uniform(0.1, 0.3)
    
    # Then solve the ode using code from Q2.1
    params = [beta, gamma, rho]
    max_time = 200
    time_steps = 1
    t, results = solve_ode(init, params, max_time, time_steps)
    
    # Return final R(T)
    S_vax_t, S_unvax_t, I_vax_t, I_unvax_t, R_vax_t, R_unvax_t = results
    Rt = R_vax_t + R_unvax_t
    return Rt

def cost_function(arm_no: float) -> float:
    """
    Return the cost function for the arm by running one simulation of stochastic_model_oracle
    Args:
        arm_no: arm number from 0-9
    Returns:
        cost: cost of running the arm
    """
    assert arm_no in list(range(10))
    # CODE HERE
    # Compute initial fractions based on the arm
    k = arm_no / 10
    susceptible = 0.9
    infected = 0.1
    
    S_vax0 = k * susceptible
    S_unvax0 = (1 - k) * susceptible
    I_vax0 = k * infected
    I_unvax0 = (1 - k) * infected
    R_vax0 = 0
    R_unvax0 = 0
    max_time = 200
    time_steps = 1
    
    init = [S_vax0, S_unvax0, I_vax0, I_unvax0, R_vax0, R_unvax0]
    
    # Pass it to stochastic_model_oracle and get R(T)
    Rt = stochastic_model_oracle(init)
    
    # Compute and return the reward
    '''ASK IF WE WANT TO USE THE MEAN OF RT ACROSS ALL TIME STEPS'''
    '''Change Rt to Rt at the last time step 200'''
    '''CHANGE THIS'''
    cost = 8 * (S_vax0 + I_unvax0) + 10 * np.mean(Rt)
    
    return cost
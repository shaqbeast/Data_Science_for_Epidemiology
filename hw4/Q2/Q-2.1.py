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

    
beta = 0.1
gamma = 0.01
rho = 0.3
'''CHANGE K VALUE HERE'''
k = 0.1 
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
params = [beta, gamma, rho]

t, results = solve_ode(init, params, max_time, time_steps)

S_vax_t, S_unvax_t, I_vax_t, I_unvax_t, R_vax_t, R_unvax_t = results

Rt_at_200 = R_vax_t[199] + R_unvax_t[199]

print(Rt_at_200)

plt.plot(t, S_unvax_t, label='Susceptible Unvaccinated')
plt.plot(t, I_unvax_t, label='Infected Unvaccinated')
plt.plot(t, R_unvax_t, label='Recovered Unvaccinated')
plt.plot(t, S_vax_t, label='Susceptible Vaccinated')
plt.plot(t, I_vax_t, label='Infected Vaccinated')
plt.plot(t, R_vax_t, label='Recovered Vaccinated')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.legend()
plt.title('SIR Model (β = ' + str(beta) + ', γ = ' + str(gamma) + 
          ', ρ =' + str(rho) + ', k = ' + str(k) + ')')
plt.show()


